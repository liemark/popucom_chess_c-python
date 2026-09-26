#include "puct.h"
#include "mcts_manager.h"
#include "mcts_search.h"
#include "game.h"
#include <memory>
#include <algorithm>

// C-style API Implementations
extern "C" {

    // This constant defines the default size for the transposition table
    // when created from the Python side, which doesn't specify a size.
    const size_t DEFAULT_TT_MAX_SIZE = 10000;

    API void* create_mcts_manager(int num_games, bool enable_noise, double initial_fpu) {
        // Correctly call the 4-argument constructor defined in mcts_manager.h/cpp
        return new MCTSManager(num_games, enable_noise, initial_fpu, DEFAULT_TT_MAX_SIZE);
    }

    API void destroy_mcts_manager(void* manager_ptr) {
        delete static_cast<MCTSManager*>(manager_ptr);
    }

    API int mcts_run_simulations_and_get_requests(void* manager_ptr, Board* board_requests_buffer, int* request_indices_buffer, int max_requests) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);

        manager->pending_requests.clear();
        int requests_count = 0;

        for (int i = 0; i < manager->num_games; ++i) {
            if (requests_count >= max_requests) break;
            if (get_game_result(&manager->game_boards[i]) != IN_PROGRESS) continue;

            Board& current_board_state = manager->game_boards[i];
            // [修改] 置换表键含贴目：不同贴目的搜索树相互隔离
            TTKey tt_key(current_board_state, manager->game_komi[i]);

            // FIFO Cache replacement logic for the transposition table
            if (manager->transposition_table.find(tt_key) == manager->transposition_table.end()) {
                if (manager->transposition_table.size() >= manager->max_tt_size) {
                    TTKey oldest_key = manager->tt_insertion_order.front();
                    manager->tt_insertion_order.pop();
                    manager->transposition_table.erase(oldest_key);
                }
                manager->transposition_table[tt_key] = std::make_shared<MCTSSearch>(&current_board_state, manager->enable_noise, manager->initial_fpu, manager->game_komi[i]);
                manager->tt_insertion_order.push(tt_key);
            }

            auto search = manager->transposition_table.at(tt_key);

            if (search->pending_evaluation_leaf_idx == INVALID_INDEX) {
                Board leaf_board;
                search->run_simulation(leaf_board);
                if (search->pending_evaluation_leaf_idx != INVALID_INDEX) {
                    board_requests_buffer[requests_count] = leaf_board;
                    request_indices_buffer[requests_count] = i;
                    manager->pending_requests.push_back({ leaf_board, search });
                    requests_count++;
                }
            }
        }
        return requests_count;
    }

    API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values, const Board* boards) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);

        for (size_t i = 0; i < manager->pending_requests.size(); ++i) {
            auto& request = manager->pending_requests[i];
            const Board& leaf_board = request.first;
            std::shared_ptr<MCTSSearch>& search = request.second;

            const float* policy_ptr = &policies[i * BOARD_SQUARES];
            float value = values[i];

            // [修改] 旧接口兼容：中性反传（weight=1、σ=0），行为等价于旧版 DLL
            search->expand_and_evaluate_neutral(leaf_board, policy_ptr, value);
        }
        manager->pending_requests.clear();
    }

    API void mcts_feed_results_with_uncertainty(void* manager_ptr, const float* policies, const float* values, const float* uncertainties, const Board* boards) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);

        for (size_t i = 0; i < manager->pending_requests.size(); ++i) {
            auto& request = manager->pending_requests[i];
            const Board& leaf_board = request.first;
            std::shared_ptr<MCTSSearch>& search = request.second;

            const float* policy_ptr = &policies[i * BOARD_SQUARES];
            float value = values[i];
            float uncertainty = uncertainties[i]; // [新增] NN 预测的不确定度

#ifdef UNCERTAINTY_SENTINEL
            // [仅 compat DLL] 负值哨兵：该请求来自不带不确定度头的旧模型，走中性反传
            // （weight=1、σ=0），与 mcts_feed_results 旧行为完全等价。
            // 仅用于新旧模型混合对打测试：build_compat.ps1 编译 popucom_core_compat.dll，
            // arena_vs_old.py 使用。主 DLL 不定义此宏，热路径零额外判断
            if (uncertainty < 0.0f) {
                search->expand_and_evaluate_neutral(leaf_board, policy_ptr, value);
                continue;
            }
#endif
            search->expand_and_evaluate(leaf_board, policy_ptr, value, uncertainty);
        }
        manager->pending_requests.clear();
    }

    API void mcts_make_move(void* manager_ptr, int game_index, int square) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return;
        std::lock_guard<std::mutex> lock(manager->mtx);

        make_move(&manager->game_boards[game_index], square);
    }

    API bool mcts_get_policy(void* manager_ptr, int game_index, float* policy_buffer) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return false;
        std::lock_guard<std::mutex> lock(manager->mtx);

        TTKey key(manager->game_boards[game_index], manager->game_komi[game_index]);
        if (manager->transposition_table.count(key)) {
            manager->transposition_table.at(key)->get_policy(policy_buffer);
            return true;
        }
        return false;
    }

    API int mcts_get_simulations_done(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);

        TTKey key(manager->game_boards[game_index], manager->game_komi[game_index]);
        if (manager->transposition_table.count(key)) {
            return manager->transposition_table.at(key)->get_simulations_done();
        }
        return 0;
    }

    API const Board* mcts_get_board_state(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return nullptr;
        return &manager->game_boards[game_index];
    }

    API void mcts_set_fpu(void* manager_ptr, double new_fpu) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->initial_fpu = new_fpu;
    }

    API void mcts_set_noise_enabled(void* manager_ptr, bool enable) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->enable_noise = enable;
    }

    API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->game_boards[game_index] = *board;
    }

    API bool mcts_is_game_over(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return true;
        std::lock_guard<std::mutex> lock(manager->mtx);
        return get_game_result(&manager->game_boards[game_index]) != IN_PROGRESS;
    }

    API void mcts_set_komi(void* manager_ptr, int game_index, int komi) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        // [新增] 设置本局贴目（白给黑）。若中途修改，旧贴目的搜索树因键不同自然失效，
        // 下一次 run 会以新贴目重新建树（旧树等待 FIFO 逐出）。
        manager->game_komi[game_index] = komi;
    }

    API int mcts_get_score_diff(void* manager_ptr, int game_index) {
        // [新增] 返回终局地板数差（黑视角，未含贴目）。
        // Python 侧据此记录目差统计，驱动自适应贴目采样；贴目调整后的胜负
        // 由 Python 自行计算（score_diff + komi > 0），或用 mcts_get_final_value。
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return 0;
        return get_score_diff(&manager->game_boards[game_index]);
    }

    API float mcts_get_final_value(void* manager_ptr, int game_index, int player_perspective) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return 0.0f;

        const Board* board = &manager->game_boards[game_index];
        // [修改] 贴目感知：双方步数用尽时 score_diff + komi 判胜负（可翻转/平局）；
        // 当前方无合法步判负与贴目无关。komi=0 时与旧行为完全等价。
        float value = 0.0f;
        if (board->moves_left[BLACK] <= 0 && board->moves_left[WHITE] <= 0) {
            int adjusted_diff = get_score_diff(board) + manager->game_komi[game_index];
            value = (adjusted_diff > 0) ? 1.0f : ((adjusted_diff < 0) ? -1.0f : 0.0f);
        }
        else {
            GameResult result = get_game_result(board);
            if (result == BLACK_WIN) value = 1.0f;
            else if (result == WHITE_WIN) value = -1.0f;
        }

        return (player_perspective == BLACK) ? value : -value;
    }

    API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);

        TTKey key(manager->game_boards[game_index], manager->game_komi[game_index]);
        if (manager->transposition_table.count(key)) {
            auto search = manager->transposition_table.at(key);
            const Node& root_node = search->node_store[search->root_idx];
            if (!root_node.is_expanded) return 0;

            int count = 0;
            double parent_visits = root_node.visit_count; // [修改] 加权访问次数，改用 double
            for (int i = 0; i < root_node.num_children; ++i) {
                if (count >= buffer_size) break;
                const Node& child = search->node_store[root_node.children_start_idx + i];
                moves_buffer[count] = child.move_leading_to_this_node;
                q_values_buffer[count] = (child.visit_count > 0.0) ? static_cast<float>(child.get_q_value()) : 0.0f;
                visit_counts_buffer[count] = static_cast<int>(child.visit_count); // [修改] 加权访问次数截断为 int 展示
                puct_scores_buffer[count] = static_cast<float>(child.get_puct_value(parent_visits, search->fpu_value));
                count++;
            }
            return count;
        }
        return 0;
    }

    API int mcts_get_child_variances(void* manager_ptr, int game_index, float* variance_buffer, int buffer_size) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);

        TTKey key(manager->game_boards[game_index], manager->game_komi[game_index]);
        if (manager->transposition_table.count(key)) {
            return manager->transposition_table.at(key)->get_child_variances(variance_buffer, buffer_size);
        }
        return 0;
    }

    API void mcts_get_legal_moves_mask(void* manager_ptr, int game_index, float* mask_buffer) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->num_games) {
            std::fill(mask_buffer, mask_buffer + BOARD_SQUARES, 0.0f);
            return;
        }
        const Board* board = &manager->game_boards[game_index];
        Bitboards legal_moves_bb = get_legal_moves(board);
        for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
            mask_buffer[sq] = GET_BIT(legal_moves_bb, sq) ? 1.0f : 0.0f;
        }
    }

    // [新增] 共用实现：填充前 11 个基础平面（局面信息，不含贴目）
    static void fill_base_planes(const Board& board, float* current_tensor_ptr) {
        const int plane_size = BOARD_SQUARES;
        const int MAX_MOVES_PER_PLAYER = 25;
        int p = board.current_player;
        int o = 1 - p;
        auto fill_plane = [&](int plane_idx, const Bitboards& bb) {
            float* plane_ptr = current_tensor_ptr + plane_idx * plane_size;
            for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
                plane_ptr[sq] = GET_BIT(bb, sq) ? 1.0f : 0.0f;
            }
            };
        auto fill_scalar_plane = [&](int plane_idx, float value) {
            float* plane_ptr = current_tensor_ptr + plane_idx * plane_size;
            std::fill(plane_ptr, plane_ptr + plane_size, value);
            };
        fill_plane(0, board.pieces[p]);
        fill_plane(1, board.pieces[o]);
        fill_plane(2, board.tiles[p]);
        fill_plane(3, board.tiles[o]);
        fill_scalar_plane(4, (p == BLACK) ? 1.0f : 0.0f);
        fill_scalar_plane(5, (p == WHITE) ? 1.0f : 0.0f);
        fill_scalar_plane(6, static_cast<float>(board.moves_left[BLACK]) / MAX_MOVES_PER_PLAYER);
        fill_scalar_plane(7, static_cast<float>(board.moves_left[WHITE]) / MAX_MOVES_PER_PLAYER);
        fill_scalar_plane(8, static_cast<float>(pop_count(&board.tiles[BLACK])) / BOARD_SQUARES);
        fill_scalar_plane(9, static_cast<float>(pop_count(&board.tiles[WHITE])) / BOARD_SQUARES);
        Bitboards all_tiles;
        all_tiles.parts[0] = ~(board.tiles[BLACK].parts[0] | board.tiles[WHITE].parts[0]);
        all_tiles.parts[1] = ~(board.tiles[BLACK].parts[1] | board.tiles[WHITE].parts[1]);
        fill_plane(10, all_tiles);
    }

    // [保留] 旧 11 通道接口：供 GUI / arena 等尚未迁移的调用方继续使用
    API void boards_to_tensors_c(const Board* boards, int num_boards, float* output_tensor) {
        const int plane_size = BOARD_SQUARES;
        const int NUM_INPUT_CHANNELS = 11;
        const int tensor_size = NUM_INPUT_CHANNELS * plane_size;
        for (int i = 0; i < num_boards; ++i) {
            fill_base_planes(boards[i], output_tensor + i * tensor_size);
        }
    }

    // [新增] 12 通道接口：通道 11 = 贴目平面（komi / KOMI_SCALE）
    // komis 为每个 board 的贴目（白给黑的目数，0 = 无贴目）
    // KOMI_SCALE 与 Python popucom_nn_interface.KOMI_SCALE 保持一致
    API void boards_to_tensors_with_komi_c(const Board* boards, int num_boards, const int* komis, float* output_tensor) {
        const int plane_size = BOARD_SQUARES;
        const int NUM_INPUT_CHANNELS_WITH_KOMI = 12;
        const float KOMI_SCALE = 20.0f;
        const int tensor_size = NUM_INPUT_CHANNELS_WITH_KOMI * plane_size;
        for (int i = 0; i < num_boards; ++i) {
            float* current_tensor_ptr = output_tensor + i * tensor_size;
            fill_base_planes(boards[i], current_tensor_ptr);
            float komi_value = static_cast<float>(komis[i]) / KOMI_SCALE;
            std::fill(current_tensor_ptr + 11 * plane_size,
                      current_tensor_ptr + 12 * plane_size, komi_value);
        }
    }
}

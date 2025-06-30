#ifndef PUCT_H
#define PUCT_H

#include "game.h"
#include <stdbool.h> // 确保 bool 类型可用

#ifdef __cplusplus
extern "C" {
#endif

	// ... (其他函数声明保持不变) ...

	/**
	 * @brief 创建一个新的 MCTS 管理器实例。
	 * @param num_games 要同时管理的游戏数量。
	 * @param enable_noise bool值，指示是否在训练时为新游戏启用狄利克雷噪声。
	 * @return 指向新创建的管理器的指针。
	 */
	API void* create_mcts_manager(int num_games, bool enable_noise);

	/**
	 * @brief 动态地为所有游戏启用或禁用狄利克雷噪声。
	 * @param manager_ptr 指向 MCTS 管理器的指针。
	 * @param enable bool值，true表示启用噪声，false表示禁用。
	 */
	API void mcts_set_noise_enabled(void* manager_ptr, bool enable);

	/**
	 * @brief (仅用于GUI/分析) 获取根节点下所有合法走棋的详细信息。
	 * (此函数声明及其他原有声明保持不变)
	 */
	API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size);

	/**
	 * @brief (仅用于GUI/分析) 为GUI重置指定游戏的MCTS状态。
	 * (此函数声明及其他原有声明保持不变)
	 */
	API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board);

	
#ifdef __cplusplus
}
#endif

#endif // PUCT_H
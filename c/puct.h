#ifndef PUCT_H
#define PUCT_H

#include "game.h" // 包含游戏核心定义

// Forward declaration for the memory pool
class Arena;

#ifdef __cplusplus
extern "C" {
#endif

	// ... (其他函数声明保持不变) ...

	/**
	 * @brief (仅用于GUI/分析) 获取根节点下所有合法走棋的详细信息。
	 * * @param manager_ptr 指向 MCTS 管理器的指针。
	 * @param game_index 要分析的游戏索引。
	 * @param moves_buffer int数组，填充合法走棋的位置。
	 * @param q_values_buffer float数组，填充对应走棋的Q值(当前玩家视角)。
	 * @param visit_counts_buffer int数组，填充对应走棋的访问次数。
	 * @param puct_scores_buffer float数组，填充对应走棋的PUCT分数。 // NEW: Added PUCT scores buffer
	 * @param buffer_size 提供的缓冲区的大小。
	 * @return int 返回找到的合法走棋数量。
	 */
	API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size);


	/**
	 * @brief (仅用于GUI/分析) 为GUI重置指定游戏的MCTS状态。
	 * * 当用户在GUI中悔棋或开始新分析时，需要用新的棋盘状态重置MCTS。
	 *
	 * @param manager_ptr 指向 MCTS 管理器的指针。
	 * @param game_index 要重置的游戏索引 (在GUI中通常为0)。
	 * @param board 指向新棋盘状态的指针。
	 */
	API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board);

#ifdef __cplusplus
}
#endif

#endif // PUCT_H

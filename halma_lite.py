#!/usr/bin/env python3
"""halma-lite: 极简跳棋(Halma)。

8x8 棋盘，每方 6 枚棋子，从一角走到对角。
走法：八方向走一格；或跳过相邻任意棋子落到空位，可连跳；
进过对方营地的棋子不许再出来。纯标准库。
"""

import argparse
import random
import sys

SIZE = 8
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1)]
PIECES = 6
GLYPH = {0: "·", 1: "●", 2: "○"}
PNAME = {0: "黑方(●)", 1: "白方(○)"}


def in_board(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def start_camp(player):
    """player 0 黑方在左上，player 1 白方在右下。"""
    if player == 0:
        return {(r, c) for r in range(SIZE) for c in range(SIZE) if r + c <= 2}
    return {(r, c) for r in range(SIZE) for c in range(SIZE) if r + c >= 12}


def goal_camp(player):
    return start_camp(1 - player)


def new_board():
    board = [[0] * SIZE for _ in range(SIZE)]
    for r, c in start_camp(0):
        board[r][c] = 1
    for r, c in start_camp(1):
        board[r][c] = 2
    return board


def pieces_of(board, player):
    mark = player + 1
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == mark]


def piece_moves(board, r, c, player):
    """返回该子的所有合法走法，每条是落点序列 [(r0,c0),...,(rn,cn)]。

    单步只能走一次；跳跃可以连跳。已进目标营的棋子只能在营内走。
    """
    camp = goal_camp(player)
    started_in_camp = (r, c) in camp
    moves = []
    # 单步：八方向相邻空位
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if in_board(nr, nc) and board[nr][nc] == 0:
            if not (started_in_camp and (nr, nc) not in camp):
                moves.append([(r, c), (nr, nc)])
    # 跳跃链：BFS 枚举所有连跳路径
    seen = {(r, c)}
    queue = [((r, c), [(r, c)])]
    while queue:
        (cr, cc), path = queue.pop(0)
        for dr, dc in DIRS:
            mr, mc = cr + dr, cc + dc
            lr, lc = cr + 2 * dr, cc + 2 * dc
            if (in_board(lr, lc) and in_board(mr, mc)
                    and board[mr][mc] != 0 and board[lr][lc] == 0
                    and (lr, lc) not in seen):
                if started_in_camp and (lr, lc) not in camp:
                    continue
                seen.add((lr, lc))
                npath = path + [(lr, lc)]
                moves.append(npath)
                queue.append(((lr, lc), npath))
    return moves


def all_moves(board, player):
    out = []
    for r, c in pieces_of(board, player):
        for m in piece_moves(board, r, c, player):
            out.append(m)
    return out


def apply_move(board, player, path):
    """在 board 上执行走法；非法则抛 ValueError。返回落点。"""
    mark = player + 1
    if not path or len(path) < 2:
        raise ValueError("走法至少需要起点和落点")
    r0, c0 = path[0]
    if not in_board(r0, c0) or board[r0][c0] != mark:
        raise ValueError("起点没有你的棋子")
    camp = goal_camp(player)
    started_in_camp = (r0, c0) in camp
    cur = (r0, c0)
    for i, nxt in enumerate(path[1:]):
        nr, nc = nxt
        if not in_board(nr, nc):
            raise ValueError("落点超出棋盘")
        dr, dc = nr - cur[0], nc - cur[1]
        adr, adc = abs(dr), abs(dc)
        is_step = max(adr, adc) == 1 and adr <= 1 and adc <= 1
        is_hop = (adr, adc) in ((2, 0), (0, 2), (2, 2))
        if i == 0:
            if is_step:
                if board[nr][nc] != 0:
                    raise ValueError("单步落点被占用")
            elif is_hop:
                mr, mc = (cur[0] + nr) // 2, (cur[1] + nc) // 2
                if board[mr][mc] == 0:
                    raise ValueError("跳跃中间没有棋子可跳")
                if board[nr][nc] != 0:
                    raise ValueError("跳跃落点被占用")
            else:
                raise ValueError("第一步必须是相邻单步或跳一子")
        else:
            if not is_hop:
                raise ValueError("连跳中只能继续跳跃")
            mr, mc = (cur[0] + nr) // 2, (cur[1] + nc) // 2
            if board[mr][mc] == 0:
                raise ValueError("跳跃中间没有棋子可跳")
            if board[nr][nc] != 0:
                raise ValueError("跳跃落点被占用")
        if started_in_camp and (nr, nc) not in camp:
            raise ValueError("已进营的棋子不能离开营地")
        cur = (nr, nc)
    board[r0][c0] = 0
    board[cur[0]][cur[1]] = mark
    return cur


def has_won(board, player):
    camp = goal_camp(player)
    mark = player + 1
    count = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == mark:
                count += 1
                if (r, c) not in camp:
                    return False
    return count == PIECES


# ---- AI：贪心最小化棋子到目标营的总距离 ----

def _assign_distance(board, player):
    """贪心分配目标营格，返回总 Chebyshev 距离。"""
    pcs = pieces_of(board, player)
    cells = list(goal_camp(player))
    total = 0
    for p in pcs:
        bi = min(range(len(cells)),
                 key=lambda i: max(abs(p[0] - cells[i][0]), abs(p[1] - cells[i][1])))
        total += max(abs(p[0] - cells[bi][0]), abs(p[1] - cells[bi][1]))
        cells.pop(bi)
    return total


def choose_move(board, player, rng):
    """贪心选走法：总距离最小；进营优先；随机打破平局。"""
    moves = all_moves(board, player)
    if not moves:
        return None
    camp = goal_camp(player)
    best, best_key = None, None
    for m in moves:
        b2 = [row[:] for row in board]
        apply_move(b2, player, m)
        d = _assign_distance(b2, player)
        enters = m[-1] in camp and m[0] not in camp
        key = (d, 0 if enters else 1, rng.random())
        if best_key is None or key < best_key:
            best, best_key = m, key
    return best


def play_game(seed=None, max_plies=600, verbose=False):
    """AI 对 AI 下一局。返回 (winner, plies)：winner 为 0/1，和棋为 None。"""
    rng = random.Random(seed)
    board = new_board()
    player = 0
    for ply in range(max_plies):
        m = choose_move(board, player, rng)
        if m is None:
            return None, ply
        apply_move(board, player, m)
        if verbose:
            print(f"第 {ply + 1} 手 {PNAME[player]}: {_fmt_path(m)}")
        if has_won(board, player):
            return player, ply + 1
        player = 1 - player
    return None, max_plies


def auto_demo(games, seed=None, max_plies=600, verbose=False):
    rng = random.Random(seed)
    wins = {0: 0, 1: 0, None: 0}
    plies_sum = 0
    for g in range(games):
        w, plies = play_game(seed=rng.randrange(1 << 30), max_plies=max_plies)
        wins[w] += 1
        plies_sum += plies
        if verbose:
            who = "和棋" if w is None else f"{PNAME[w]}胜"
            print(f"第 {g + 1} 局：{who}，{plies} 手")
    return wins, plies_sum / games if games else 0.0


# ---- 文本界面 ----

def _fmt_cell(r, c):
    return f"{'abcdefgh'[c]}{SIZE - r}"


def _fmt_path(path):
    return "-".join(_fmt_cell(r, c) for r, c in path)


def _parse_cell(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in "abcdefgh" or s[1] not in "12345678":
        return None
    return (SIZE - int(s[1]), "abcdefgh".index(s[0]))


def _parse_path(s):
    parts = s.replace(" ", "-").split("-")
    cells = [_parse_cell(p) for p in parts]
    if any(c is None for c in cells) or len(cells) < 2:
        return None
    return cells


def render(board):
    lines = ["  " + " ".join("abcdefgh")]
    for r in range(SIZE):
        lines.append(f"{SIZE - r} " + " ".join(GLYPH[board[r][c]] for c in range(SIZE)))
    return "\n".join(lines)


def interactive(seed=None):
    if not sys.stdin.isatty():
        print("交互模式需要终端；无头演示请用 --auto", file=sys.stderr)
        return 2
    rng = random.Random(seed)
    board = new_board()
    player = 0  # 人类执黑先行
    ply = 0
    print("halma-lite：你是黑方(●)，左上开局，目标是把 6 枚棋子全部送进右下营地。")
    print("输入走法如 a8-b7-c6（单步或连跳），q 退出。")
    while True:
        print()
        print(render(board))
        if player == 0:
            s = input(f"第 {ply + 1} 手，你走: ").strip()
            if s.lower() in ("q", "quit", "exit"):
                print("已退出。")
                return 0
            path = _parse_path(s)
            if path is None:
                print("格式不对，示例：a8-b7")
                continue
            try:
                apply_move(board, 0, path)
            except ValueError as e:
                print(f"非法走法：{e}")
                continue
        else:
            m = choose_move(board, 1, rng)
            if m is None:
                print("AI 无棋可走，和棋。")
                return 0
            apply_move(board, 1, m)
            print(f"AI 走：{_fmt_path(m)}")
        ply += 1
        if has_won(board, player):
            print()
            print(render(board))
            print("你赢了！" if player == 0 else "AI 赢了。")
            return 0
        player = 1 - player


def main(argv=None):
    ap = argparse.ArgumentParser(description="halma-lite：极简跳棋，8x8，贪心 AI")
    ap.add_argument("--auto", action="store_true", help="AI 对 AI 自动演示")
    ap.add_argument("--games", type=int, default=10, help="自动演示局数（默认 10）")
    ap.add_argument("--max-plies", type=int, default=600, help="单局最大手数（默认 600）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--verbose", action="store_true", help="打印每手/每局")
    args = ap.parse_args(argv)
    if args.auto:
        wins, avg = auto_demo(args.games, seed=args.seed,
                              max_plies=args.max_plies, verbose=args.verbose)
        print(f"自动演示结束：共 {args.games} 局，"
              f"黑方胜 {wins[0]}，白方胜 {wins[1]}，和棋 {wins[None]}，"
              f"平均 {avg:.1f} 手/局")
        return 0
    return interactive(seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())

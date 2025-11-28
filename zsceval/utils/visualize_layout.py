#!/usr/bin/env python3
"""
Overcooked Layout Visualizer for overcooked_new
레이아웃 파일을 이미지로 저장하는 간단한 스크립트
"""

import argparse
import os
from pathlib import Path

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
)
from zsceval.envs.overcooked_new.src.overcooked_ai_py.utils import load_dict_from_file
from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer,
)

# 사용할 수 있는 셰프 모자 색상
AVAILABLE_CHEF_COLORS = ["blue", "green", "red", "purple", "orange"]


def parse_grid_lines(grid_str):
    """레이아웃의 grid 문자열을 라인 리스트로 파싱"""
    grid = grid_str.strip()
    return [line.strip() for line in grid.split("\n") if line.strip()]


def render_layout_to_image(layout_file_path, output_path, show_players=False, tile_size=75):
    """
    레이아웃 파일을 이미지로 렌더링하여 저장
    
    Args:
        layout_file_path: .layout 파일 경로
        output_path: 출력 이미지 파일 경로 (.png)
        show_players: 플레이어를 표시할지 여부 (기본: False)
        tile_size: 타일 크기 (기본: 75)
    """
    # 레이아웃 파일 로드
    if not os.path.exists(layout_file_path):
        raise FileNotFoundError(f"Layout file not found: {layout_file_path}")
    
    layout_data = load_dict_from_file(layout_file_path)
    
    if "grid" not in layout_data:
        raise ValueError(f"Layout file does not contain 'grid' field: {layout_file_path}")
    
    # 레이아웃 이름 추출 (파일명에서)
    layout_name = Path(layout_file_path).stem
    
    # Grid 파싱
    lines = parse_grid_lines(layout_data["grid"])
    
    # OvercookedGridworld 생성
    # grid를 제외한 나머지 파라미터들을 base_layout_params로 전달
    base_layout_params = {
        "layout_name": layout_name,
        **{k: v for k, v in layout_data.items() if k != "grid"},
    }
    mdp = OvercookedGridworld.from_grid(lines, base_layout_params=base_layout_params)
    
    # State 생성
    if show_players:
        state = mdp.get_standard_start_state()
    else:
        # 플레이어 없이 빈 state 생성
        state = OvercookedState(
            players=[],
            objects={},
            bonus_orders=mdp.start_bonus_orders,
            all_orders=mdp.start_all_orders,
        )
    
    # 플레이어 수만큼 색상 리스트 생성 (부족하면 순환 재사용)
    num_players = len(state.players) if show_players else 0
    player_colors = [
        AVAILABLE_CHEF_COLORS[i % len(AVAILABLE_CHEF_COLORS)] for i in range(num_players)
    ]
    
    # StateVisualizer 생성
    visualizer = StateVisualizer(
        tile_size=tile_size,
        is_rendering_hud=False,
        is_rendering_cooking_timer=False,
        is_rendering_action_probs=False,
        player_colors=player_colors if player_colors else None,
    )
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # 이미지 저장
    saved_path = visualizer.display_rendered_state(
        state=state,
        grid=mdp.terrain_mtx,
        img_path=output_path,
        ipython_display=False,
        window_display=False,
    )
    
    print(f"✓ Layout image saved to: {saved_path}")
    return saved_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Overcooked Layout Visualizer - Convert .layout file to image"
    )
    parser.add_argument(
        "layout_file",
        type=str,
        help="Path to the .layout file (e.g., path/to/layout.layout)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <layout_name>.png in current directory)",
    )
    parser.add_argument(
        "--show-players",
        action="store_true",
        help="Show players in the rendered image (default: False)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=75,
        help="Tile size for rendering (default: 75)",
    )
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        layout_name = Path(args.layout_file).stem
        args.output = f"{layout_name}.png"
    
    # 레이아웃 이미지로 변환
    try:
        render_layout_to_image(
            layout_file_path=args.layout_file,
            output_path=args.output,
            show_players=args.show_players,
            tile_size=args.tile_size,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


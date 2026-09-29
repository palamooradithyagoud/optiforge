"""
src/nsga2/convergence.py
Convergence tracking, 2D hypervolume calculation, and convergence plot generation.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.nsga2.population import Individual


def compute_hypervolume_2d(points: List[Tuple[float, float]], ref_point: Tuple[float, float] = (1.5, 2.0)) -> float:
    """
    Compute exact 2D hypervolume dominated by a set of non-dominated points
    with respect to a reference nadir point (r_x, r_y) in minimization space.
    
    Formula:
    Filter points bounded by ref_point.
    Sort points by x ascending.
    Compute union of non-overlapping rectangular slices.
    """
    rx, ry = ref_point
    valid_pts = [(x, y) for (x, y) in points if x <= rx and y <= ry]
    if not valid_pts:
        return 0.0
        
    # Sort by x ascending
    valid_pts.sort(key=lambda p: (p[0], p[1]))
    
    # Remove dominated points within 2D projection
    non_dom = []
    min_y = float("inf")
    for x, y in valid_pts:
        if y < min_y:
            non_dom.append((x, y))
            min_y = y
            
    # Calculate hypervolume as sum of rectangular areas
    hv = 0.0
    for i in range(len(non_dom)):
        x_curr, y_curr = non_dom[i]
        x_next = non_dom[i + 1][0] if (i + 1 < len(non_dom)) else rx
        width = x_next - x_curr
        height = ry - y_curr
        if width > 0 and height > 0:
            hv += width * height
            
    return float(round(hv, 6))


class ConvergenceTracker:
    """Tracks generation-by-generation evolution metrics and exports convergence artifacts."""
    def __init__(self, ref_point: Tuple[float, float] = (1.5, 2.0)):
        self.ref_point = ref_point
        self.history: List[Dict[str, Any]] = []

    def record_generation(
        self,
        generation: int,
        population: List[Individual],
        pareto_front: List[Individual],
        elapsed_sec: float
    ) -> Dict[str, Any]:
        """Record metrics for current generation."""
        maes = [ind.metrics.validation_mae for ind in population if ind.metrics]
        gaps = [ind.metrics.generalization_gap for ind in population if ind.metrics]
        params = [ind.metrics.trainable_parameters for ind in population if ind.metrics]
        latencies = [ind.metrics.inference_latency_ms for ind in population if ind.metrics]
        
        # Hypervolume on Pareto front (f1: MAE, f2: generalization_gap)
        pf_pts = [(ind.metrics.validation_mae, ind.metrics.generalization_gap) for ind in pareto_front if ind.metrics]
        hv = compute_hypervolume_2d(pf_pts, ref_point=self.ref_point)
        
        entry = {
            "generation": generation,
            "population_size": len(population),
            "pareto_front_size": len(pareto_front),
            "best_mae": round(float(np.min(maes)), 4) if maes else 0.0,
            "mean_mae": round(float(np.mean(maes)), 4) if maes else 0.0,
            "best_generalization_gap": round(float(np.min(gaps)), 4) if gaps else 0.0,
            "min_parameter_count": int(np.min(params)) if params else 0,
            "min_latency_ms": round(float(np.min(latencies)), 4) if latencies else 0.0,
            "hypervolume_f1_f2": round(hv, 5),
            "generation_time_sec": round(float(elapsed_sec), 3)
        }
        self.history.append(entry)
        return entry

    def save_csv(self, filepath: str) -> pd.DataFrame:
        """Save convergence history to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        return df

    def generate_plot(self, filepath: str):
        """Render publication-quality convergence plot using PIL."""
        if not self.history:
            return
            
        df = pd.DataFrame(self.history)
        w, h = 900, 520
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Margins
        ml, mr, mt, mb = 70, 50, 60, 60
        pw = w - ml - mr
        ph = h - mt - mb
        
        n_gen = len(df)
        if n_gen <= 1:
            draw.text((ml, mt), "Single generation run - insufficient points for plot", fill=(50, 50, 50))
            img.save(filepath)
            return
            
        # Draw axes
        draw.line([(ml, mt), (ml, mt + ph)], fill=(120, 120, 120), width=2)
        draw.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=(120, 120, 120), width=2)
        
        # Metric 1: Best MAE (left scale)
        mae_min = max(0.5, df["best_mae"].min() * 0.9)
        mae_max = df["best_mae"].max() * 1.05
        
        def gen_to_x(g):
            return ml + (g / (n_gen - 1)) * pw
            
        def mae_to_y(m):
            norm = (m - mae_min) / max((mae_max - mae_min), 1e-4)
            return mt + ph - (norm * ph)
            
        # Draw grid
        for i in range(5):
            val = mae_min + (i / 4.0) * (mae_max - mae_min)
            y_pos = mae_to_y(val)
            draw.line([(ml, y_pos), (ml + pw, y_pos)], fill=(235, 235, 235), width=1)
            draw.text((ml - 55, y_pos - 6), f"{val:.2f}", fill=(80, 80, 80))
            
        for g in range(0, n_gen, max(1, n_gen // 6)):
            x_pos = gen_to_x(g)
            draw.line([(x_pos, mt), (x_pos, mt + ph)], fill=(240, 240, 240), width=1)
            draw.text((x_pos - 6, mt + ph + 8), str(g), fill=(80, 80, 80))
            
        # Line 1: Best MAE (Blue)
        mae_pts = [(gen_to_x(row["generation"]), mae_to_y(row["best_mae"])) for _, row in df.iterrows()]
        for i in range(len(mae_pts) - 1):
            draw.line([mae_pts[i], mae_pts[i + 1]], fill=(31, 119, 180), width=3)
            
        # Line 2: Mean MAE (Orange)
        mean_mae_pts = [(gen_to_x(row["generation"]), mae_to_y(row["mean_mae"])) for _, row in df.iterrows()]
        for i in range(len(mean_mae_pts) - 1):
            draw.line([mean_mae_pts[i], mean_mae_pts[i + 1]], fill=(255, 127, 14), width=2)
            
        # Title and Labels
        draw.text((ml + 120, 20), "NSGA-II Multi-Objective Evolutionary Convergence", fill=(20, 20, 20))
        draw.text((ml + pw // 2 - 40, mt + ph + 30), "Generation", fill=(40, 40, 40))
        
        # Legend
        draw.line([(w - 240, 25), (w - 200, 25)], fill=(31, 119, 180), width=3)
        draw.text((w - 190, 20), "Best Val MAE", fill=(31, 119, 180))
        draw.line([(w - 240, 45), (w - 200, 45)], fill=(255, 127, 14), width=2)
        draw.text((w - 190, 40), "Mean Val MAE", fill=(255, 127, 14))
        
        img.save(filepath)

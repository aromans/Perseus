import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

epochs = list(range(1, 31))

train_loss = [
    0.7678, 0.5701, 0.4714, 0.3039, 0.1846,
    0.1601, 0.1243, 0.1054, 0.05863, 0.04086,
    0.03766, 0.03065, 0.03280, 0.02999, 0.01357,
    0.01121, 0.009043, 0.01287, 0.007048, 0.006301,
    0.009348, 0.005598, 0.006376, 0.005481, 0.003932,
    0.00594, 0.00444, 0.005445, 0.004314, 0.00462,
]

eval_loss = [
    0.7078, 0.5696, 0.4207, 0.3284, 0.3115,
    0.3258, 0.3463, 0.3481, 0.4195, 0.4155,
    0.4433, 0.4127, 0.4583, 0.4868, 0.5091,
    0.5357, 0.5586, 0.5783, 0.5866, 0.5865,
    0.5897, 0.5933, 0.5973, 0.6007, 0.6039,
    0.6054, 0.6061, 0.6064, 0.6065, 0.6066,
]

eval_accuracy = [
    0.8307, 0.8586, 0.8829, 0.9052, 0.9115,
    0.9115, 0.9145, 0.9174, 0.9122, 0.9163,
    0.9137, 0.9120, 0.9146, 0.9154, 0.9150,
    0.9154, 0.9145, 0.9153, 0.9156, 0.9151,
    0.9143, 0.9151, 0.9151, 0.9153, 0.9153,
    0.9154, 0.9154, 0.9154, 0.9153, 0.9153,
]

best_epoch = eval_loss.index(min(eval_loss)) + 1  # epoch 5

fig = plt.figure(figsize=(14, 9))
fig.suptitle('Perseus Session 1 — Qwen2.5-Coder-1.5B LoRA Fine-Tune\n(15 train / 6 val examples, 30 epochs, CPU)', fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# --- Plot 1: Train vs Val Loss ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(epochs, train_loss, 'o-', color='steelblue', linewidth=2, markersize=4, label='Train Loss')
ax1.plot(epochs, eval_loss, 's-', color='tomato', linewidth=2, markersize=4, label='Val Loss')
ax1.axvline(best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Best Val Loss (epoch {best_epoch})')
ax1.axvspan(best_epoch, 30, alpha=0.05, color='red', label='Overfitting zone')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Train Loss vs Validation Loss')
ax1.legend(fontsize=9)
ax1.set_xlim(1, 30)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.annotate(f'Best: {min(eval_loss):.4f}', xy=(best_epoch, min(eval_loss)),
             xytext=(best_epoch + 2, min(eval_loss) + 0.05),
             arrowprops=dict(arrowstyle='->', color='green'),
             color='green', fontsize=9)

# --- Plot 2: Val Accuracy ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(epochs, [a * 100 for a in eval_accuracy], 's-', color='tomato', linewidth=2, markersize=4)
ax2.axvline(best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Token Accuracy (%)')
ax2.set_title('Validation Token Accuracy')
ax2.set_xlim(1, 30)
ax2.set_ylim(80, 95)
ax2.grid(True, alpha=0.3)
ax2.annotate(f'Plateau ~{max(eval_accuracy)*100:.1f}%', xy=(8, max(eval_accuracy)*100),
             xytext=(12, max(eval_accuracy)*100 - 1.5),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=9, color='gray')

# --- Plot 3: Loss Gap (overfitting indicator) ---
ax3 = fig.add_subplot(gs[1, 1])
gap = [e - t for e, t in zip(eval_loss, train_loss)]
colors = ['green' if g < 0.05 else 'orange' if g < 0.2 else 'tomato' for g in gap]
ax3.bar(epochs, gap, color=colors, alpha=0.75, edgecolor='gray', linewidth=0.5)
ax3.axvline(best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Best val epoch ({best_epoch})')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Val Loss − Train Loss')
ax3.set_title('Generalization Gap (Overfitting Indicator)')
ax3.legend(fontsize=9)
ax3.set_xlim(0.5, 30.5)
ax3.grid(True, alpha=0.3, axis='y')

out_path = '/home/perseus/Dev/Perseus/Development/session1_training_plot.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

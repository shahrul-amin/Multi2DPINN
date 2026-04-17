import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Read .geo file
file_path = 'data/test_trees/8000.geo'
with open(file_path) as f:
    lines = f.readlines()

fig, ax = plt.subplots(figsize=(15, 3))

# Colors for visualizing different segments
colors = plt.cm.tab10.colors

seg_id = 0
for line in lines:
    if line.startswith('Rectangle'):
        # Extract the parameters from {x, y, z, L, W, ...}
        rect_params = re.findall(r'-*\d+\.?\d*', line)
        if len(rect_params) >= 6:
            # According to OpenCASCADE format usually: x, y, z, dx, dy, dz
            # But the regex extracts all numbers including the Rectangle(1) index
            # So looking at the format: Rectangle(1) = {8.000000, 127.500000, 0, 24.000000, 1.000000, 0};
            # Index 0 is the ID (1)
            # Index 1 is x (8.0)
            # Index 2 is y (127.5)
            # Index 3 is z (0)
            # Index 4 is width/L (24.0)
            # Index 5 is height/W (1.0)
            
            x = float(rect_params[1])
            y = float(rect_params[2])
            width = float(rect_params[4])
            height = float(rect_params[5])
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), width, height, 
                                     linewidth=1, edgecolor='black', 
                                     facecolor=colors[seg_id % len(colors)],
                                     label=f"Seg {seg_id+1}: L={width}")
            ax.add_patch(rect)
            
            # Add text label in the middle of the segment
            ax.text(x + width/2, y + height/2, f"{seg_id+1}", 
                    ha='center', va='center', color='white', fontweight='bold')
            seg_id += 1

ax.autoscale_view()
plt.title(f"Geometry Visualization for {file_path}")
plt.xlabel("X coordinate (\mu m)")
plt.ylabel("Y coordinate (\mu m)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('wire_visualization2.png', dpi=300, bbox_inches='tight')
print("Saved visualization to wire_visualization2.png")

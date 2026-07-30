# Single-Plane Gravity Review Geometry

## Coordinate and unit conventions

- X: positive east
- Y: positive north
- Z/depth: positive downward
- Density array: `density[z, y, x]`
- Gravity array: `gravity[y, x]`
- Density contrast: g/cm3
- Gravity: vertical gravity anomaly Gz in mGal

## Density domain

- Grid shape: 24 × 64 × 64
- Cell spacing: 10 × 10 × 10 m
- X physical edges: -5 to 635 m
- Y physical edges: -5 to 635 m
- Depth physical edges: 0 to 240 m

## Observation plane

- X coordinates: -85 to 715 m
- Y coordinates: -85 to 715 m
- Z coordinate: 0 m
- Shape: 81 × 81
- Number of points: 6,561
- Spacing: 10 m
- Coordinate span: 800 m
- Figure markers: actual observation locations, displayed every 8th point in X and Y
- Gravity figure display limits (plotting only): X -85 to 715 m; Y -85 to 715 m
- Saved gravity arrays, receiver coordinates, and forward-model domain retain the complete observation extent.

## Common plotting scales

- Density scale for every plan and section: 0.0 to 1 g/cm3
- Common-scale vertical gravity anomaly Gz for every example: 1.410943435152e-05 to 7.023463660710e-01 mGal
- Each example also includes a linearly, individually scaled Gz shape/extent panel. Its colors are not amplitude-comparable between examples.

The horizontal density projection is the maximum density contrast at each X-Y location over the full depth dimension.

## Five-times criterion

- Maximum width/length/bottom depth: 160 m
- Required extent: 5 × 160 = 800 m
- Actual extent: 800 m
- Result: PASS

## Deterministic examples

| Example | Top (m) | Bottom (m) | Width X (m) | Width Y (m) | Thickness (m) | Density (g/cm³) | Center X (m) | Center Y (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| example_01_shallow_small | 20 | 40 | 40 | 40 | 20 | 0.4 | 315 | 315 |
| example_02_shallow_broad | 30 | 70 | 120 | 100 | 40 | 0.6 | 315 | 315 |
| example_03_intermediate | 50 | 100 | 80 | 100 | 50 | 0.7 | 315 | 315 |
| example_04_deeper_compact | 70 | 120 | 50 | 60 | 50 | 0.8 | 320 | 315 |
| example_05_deepest_largest | 80 | 160 | 160 | 160 | 80 | 1 | 315 | 315 |
| controlled_example_a_baseline | 20 | 50 | 60 | 60 | 30 | 0.5 | 315 | 315 |
| controlled_example_b_increased_depth | 70 | 100 | 60 | 60 | 30 | 0.5 | 315 | 315 |
| controlled_example_c_increased_horizontal_size | 20 | 50 | 140 | 140 | 30 | 0.5 | 315 | 315 |
| controlled_example_d_increased_density_contrast | 20 | 50 | 60 | 60 | 30 | 1 | 315 | 315 |
| controlled_example_e_increased_thickness | 20 | 100 | 60 | 60 | 80 | 0.5 | 315 | 315 |
| controlled_example_f_maximum_combined_case | 80 | 160 | 160 | 160 | 80 | 1 | 315 | 315 |
| controlled_example_g_left_shifted_position | 20 | 50 | 60 | 60 | 30 | 0.5 | 175 | 315 |
| controlled_example_h_right_shifted_position | 20 | 50 | 60 | 60 | 30 | 0.5 | 455 | 315 |

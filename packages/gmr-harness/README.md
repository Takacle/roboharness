# GMR-Harness

Alignment toolchain for General Motion Retargeting (GMR) + Robot Simulation.

## Installation

```bash
pip install gmr-harness
```

### Optional dependencies

```bash
pip install gmr-harness[smplx]     # SMPL-X template calibration
pip install gmr-harness[mujoco]    # MuJoCo rendering and T-pose staging
pip install gmr-harness[vlm]       # VLM-based alignment agent
pip install gmr-harness[all]       # Everything
```

### External dependency: GMR

GMR (general_motion_retargeting) is an external non-pip dependency. Setup options:

1. Clone GMR next to your project:
   ```bash
   git clone <GMR_URL> ../GMR
   ```

2. Set environment variable:
   ```bash
   export GMR_ROOT=/path/to/GMR
   ```

GMR must contain: `general_motion_retargeting/params.py`

## Usage

```bash
# Setup a new robot
gmr-harness setup --robot my_robot --xml $GMR_ROOT/assets/my_robot/robot.xml --formats smplx bvh

# Stage T-pose
gmr-harness stage --robot my_robot --preset tpose --output_dir specs/tpose/

# Validate alignment
gmr-harness validate --robot my_robot --src bvh --tpose_motion /path/to/tpose.bvh

# AI-driven alignment agent
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/motion.bvh
```

## License

MIT

# Galbot G1 Golf asset provenance

The URDF and mesh files in this directory come from the public upstream
repository:

- Repository: <https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description>
- Commit: `b311f5ca1acf506e9b7026397e2c74fb2db11df6`
- Upstream license: Apache License 2.0, copied as [`LICENSE`](LICENSE)

`galbot_one_golf.urdf` is upstream's fixed-base URDF with three compatibility
changes documented in its header: the legacy replay base datum, explicit axes
on fixed joints, and a MuJoCo compiler block.

`galbot_one_golf_with_sites.xml` was generated from that URDF with MuJoCo 3.4
using collision meshes and `fusestatic=false`, then given left/right TCP marker
sites. Preserving fixed bodies keeps wrist-camera geometry separate from the arm
overlay mask. The generated MJCF preserves the public upstream kinematics and
joint limits.

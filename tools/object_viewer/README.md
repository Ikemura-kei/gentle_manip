# Object atlas

Run from the repository root with Python 3 (standard library only):

```bash
python3 tools/object_viewer/serve.py
```

Open http://localhost:8765. Use `--port 8766` to choose another port.
The server binds to loopback. For a remote workstation, forward the port with
`ssh -L 8765:localhost:8765 user@host`.

The catalog is generated from `gentle_manip/assets/registry.py` at startup.
The default view contains nine registered food target types (eight food families
if banana and banana_chunk are grouped), including parked raspberry. Numbered
mesh variants are optional. Calibration objects, gelatin/sponge development
presets, letters, and synthetic shapes are available through the group filter.
These counts describe registered assets, not successful training or deployment.

Meshes retain their actual metric dimensions; the camera fits each selection.
The floor grid has 1 cm spacing. Display colors are illustrative, not textures.
Three.js 0.170.0 and its MIT license are bundled under vendor; no CDN connection,
Genesis, GPU simulation environment, or package installation is needed to run it.

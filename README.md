Will refine my approach following the read:
https://www.nature.com/articles/s41567-022-01788-5.epdf?sharing_token=ube1KozTYa5LaC9cu6hUTNRgN0jAjWel9jnR3ZoTv0NVOvMSmh5IwcU6Uxmom-KR2i-Pcwh_ETc6--qXhoO5LUGumCj0CT7GiaXaqsPr0FAEGtEInUThPONICh3K7Yk7QT9j7819reQGUGm4B7YVD61HSBoWLK0qWbTG__eeIcs%3D


# Fluid-Dynamics-Simulation
GPU-sided fluid dynamics simulation, with C++, Vulkan


Smoothed-particle hydrodynamics (SPH) approach based on Navier-Stokes equations for incompressible flow


![FluidDynamicsSimulation_XrIAYd1XXk](https://github.com/user-attachments/assets/45049b6a-9c97-4af6-a54c-09fa668579f8)
 

https://github.com/user-attachments/assets/38ce9239-f8ac-43ae-98b4-1470b74f8598


 # Inspiration / Reference
- https://youtu.be/XmzBREkK8kY
- https://www.youtube.com/watch?v=4b80sR-joNY
- https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
- https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics
- https://github.com/napframework/nap/tree/main/system_modules/naprender
- https://github.com/mmaldacker/Vortex2D/tree/master


## Building the Project

### Cloning the Repository

```
git clone https://github.com/in-c0/three-body-simulation.git
cd three-body-simulation
```

### Setting Up Dependencies

The following dependencies are already included in the `third_party` directory: 
- **GLFW**
- **GLM**
- **Dear ImGUI*

If you have trouble connecting to the submodules, you can manually set up the dependencies:

```
git submodule add https://github.com/glfw/glfw.git third_party/glfw
git submodule add https://github.com/g-truc/glm.git third_party/glm
git submodule add https://github.com/ocornut/imgui.git third_party/imgui
git submodule update --init --recursive
```

### Compiling and Building

First, make build files with CMAKE at the root project directory (this will create or update the files in `Fluid-Danymics-Simulation/build`):
```
cmake -B build
```

then you can trigger the build, which will create an executable inside the build folder:
```
cmake --build build
```

Now you can run the simulation with
```
./FluidDynamicsSimulation --Debug
```


## Contributing

Contributions are welcome! If you find bugs or have suggestions for improvements, feel free to create issues or submit pull requests.

1. **Fork the repository**.
2. **Create a new branch** (`git checkout -b feature-branch`).
3. **Make your changes**.
4. **Commit your changes** (`git commit -m 'Add some feature'`).
5. **Push to the branch** (`git push origin feature-branch`).
6. **Open a pull request**.


### Troubleshooting Installation

If any of the installation commands fails, ensure your system is up-to-date:

Linux:
```
 sudo apt update
 sudo apt upgrade
```
If you're not on the latest Ubuntu LTS version (e.g., 22.04 LTS), you can upgrade by:
 ```
 sudo do-release-upgrade
 ```

If this doesn't fix the issue, consider re-installing WSL and Ubuntu by following these steps:
 1. Open PowerShell or Command Prompt and run:
 3. Uninstall the current Ubuntu distribution:
```
wsl --unregister Ubuntu
```     
(or check `wsl --list --verbose` for specific distribution name to unregsiter)
 4. Reinstall Ubuntu LTS:
 ```
 wsl --install -d Ubuntu-22.04
 ```

If you prefer to work on a non-WSL/Ubuntu environment, or if you've encountered unsolvable issues during installation, you can download the prerequisite libraries from the official websites:
- [GFortran](https://fortran-lang.org/learn/os_setup/install_gfortran/)
- [CMake](https://cmake.org/download/) (version 3.10 or higher)
- [Vulkan](https://vulkan.lunarg.com/doc/sdk/1.3.290.0/linux/getting_started.html)


## License

See the [LICENSE](LICENSE) file for details.

This project has some third-party dependencies, each of which may have independent licensing:

* [glfw](https://github.com/glfw/glfw): A multi-platform library for OpenGL, OpenGL ES, Vulkan, window and input
* [glm](https://github.com/g-truc/glm): OpenGL Mathematics
* [glslang](https://github.com/KhronosGroup/glslang): Shader front end and validator
* [dear imgui](https://github.com/ocornut/imgui): Immediate Mode Graphical User Interface
* [vulkan](https://github.com/KhronosGroup/Vulkan-Docs): Sources for the formal documentation of the Vulkan API

## Acknowledgments

This project builds upon the `compute shader` sample from the [Compute Shader tutorial](https://vulkan-tutorial.com/Compute_Shader) by Alexander Overvoorde

The base implementation can be found in the [`compute_shader.cpp` file](https://vulkan-tutorial.com/code/31_compute_shader.cpp), which has been adapted to meet our project's requirements. Full credit to the original authors, and we maintain the original licensing terms as stipulated under the Apache License 2.0.

- [LunarG Vulkan SDK](https://vulkan.lunarg.com/) - Graphics API
- [GLFW](https://www.glfw.org/) - Input management
- [GLM](https://glm.g-truc.net/0.9.9/index.html) - Math utilities


## How it builds up / Feature Roadmap
How it builds up from the provided compute shader is as follows: (implemented features are marked with [x])

![image](https://github.com/user-attachments/assets/753328a1-53db-4ee7-96dc-9b2821fe79d1)


- [x] Vulkan Instance initialisation (setting up surface, debug callbacks for validation layers)
- [x] Device Selection to support for adequate queue families (graphics and compute)
- [x] Renderpass and Swapchain Creation for presenting rendered images to the screen
- [x] Graphics pipeline setup (point primitives)
- [x] Compute pipeline setup for rendering particles (including descriptor set layout)
- [x] Basic Shader setup (vert, frag, compute)
- [x] Basic Render Loop and frame synchronization (semaphores, fences)
- [x] Basic Gravity (downward push constant uniform layout)
- [x] Smoothed-particle hydrodynamics (Navier-Stokes):
- [x] ㄴ Grid-based Velocity Field
- [x] ㄴ Advection
- [x] ㄴ Diffusion
- [x] ㄴ Pressure calculation (enforcing incompressibility)
- [x] ㄴ Projection step
- [x] ㄴ Ping-pong buffering to avoid readwrite conflict
- [x] ㄴ Trilinear interpolation for advection accuracy


**Optional:**
- [x] Boundary collisions
- [ ] Event handling (mouse, keyboard inputs)
- [ ] GUI for dynamic input / user settings (Dear ImGUI)

**In radar:**
- [ ] Workgroups for parallel processing
- [ ] Memory barriers





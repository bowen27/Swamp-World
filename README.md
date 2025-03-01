# Swamp-World
# Hydrological Model Overview

This file implements a comprehensive hydrological simulation that mimics water cycle dynamics on a synthetic planetary surface. The main components are summarized below:

## 1. Initialization
- **generate_topography(phi, choice, height, half_width):**  
  Generates synthetic topography based on various profiles (e.g., Uniform, Sin, Gaussian).

- **initialize_model(params):**  
  Sets up the grid, computes latitudes in both radians and degrees, and initializes topography and water levels.

- **estimate_wetness_initial(phi, I, n_s, topo, r, K_poro):**  
  Estimates the initial wetness distribution and water level based on total water inventory and porosity.

- **calculate_evaporation(phi, topo, wet_idx, S, r):**  
  Computes evaporation rates considering solar input and local topography.

- **calculate_precipitation(phi, evap, extreme_index, spatial_assumption, ITCZ_range):**  
  Determines precipitation distribution using various spatial assumptions (local, ITCZ, decay).

## 2. Shoreline and Swamp Evolution
- **slope_evaluation(...):**  
  Evaluates shoreline stability via integrated moisture flux and slope conditions.

- **estimate_wetness_iterate(...):**  
  Iteratively updates the wetness distribution and water table height.

- **evolve_swamp(...):**  
  Adjusts wetness indices in swamp regions based on evaporation, precipitation, and groundwater flux differences.

## 3. Groundwater Evolution
- **dh_dt(h, phi, infil, flux, ...):**  
  Computes the rate of change of hydraulic head using gradient approximations.

- **rk4_integration(...):**  
  Applies a 4th order Runge-Kutta method to update the hydraulic head over time.

- **groundwater_evolution(...):**  
  Simulates groundwater redistribution on land, adjusting for boundary conditions and topographic constraints.

## 4. Diagnostics and Statistics
- **calculate_mass_conservation(...):**  
  Checks the water inventory conservation and computes the relative error.

- **calculate_groundwater_amplitude(...):**  
  Determines the amplitude of the groundwater table (difference between max and min hydraulic head).

- **calculate_surface_fractions(...):**  
  Computes latitude-weighted fractions of wet surface and swamp area.

- **calculate_cycling_times(...):**  
  Estimates both surface water and groundwater cycling times based on evaporation and groundwater flux.

## 5. Saving and Running the Model
- **save_results(...):**  
  Saves simulation results and input parameters to an HDF5 file with a filename that reflects key parameters.

- **run_hydrological_model(params, save, debug):**  
  The main driver function that:
  - Initializes the model
  - Runs shoreline stabilization and, if needed, swamp and groundwater evolution
  - Computes diagnostics (mass conservation, cycling times, etc.)
  - Optionally plots intermediate results (in debug mode)
  - Saves final results and returns a dictionary with all key outputs such as:
    - `phi`: Co-latitudes (radians)
    - `latitude`: Latitudes (degrees)
    - `topo`: Topography (m)
    - `final_head`: Final hydraulic head (m)
    - `final_wetness`: Final wetness distribution (m)
    - `final_evap`: Evaporation rates (m/s)
    - `final_precip`: Precipitation rates (m/s)
    - `land_indices` & `swamp_indices`: Indices for land and swamp regions
    - `error_I`: Mass conservation error
    - `groundwater_amplitude`: Amplitude of the groundwater table
    - `wet_surface_fraction` & `swamp_fraction`: Surface fractions (%)
    - `surface_cycling_time` & `groundwater_cycling_time`: Cycling times (years)

---

This modular structure allows the model to simulate the complex interplay between topography, evaporation, precipitation, and groundwater flow, ultimately providing a rich set of diagnostics and visualizations for analyzing hydrological behavior on a planetary scale.
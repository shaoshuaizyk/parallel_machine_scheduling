import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

def plot_machine_gantt(model, E, v, s, P, machines, jobs, depots, N, T_var):
    """
    Plot Gantt chart from the machine perspective.

    Parameters:
    model, E, v, s, P, machines, jobs, depots, N, T_var: Same as previously described.
    """
    if model.SolCount == 0:
        print("No feasible solution found; cannot plot Gantt chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    color_map = plt.cm.get_cmap("tab20", len(machines))
    dummy_start, dummy_end = 0, N + 1

    for k_idx, k in enumerate(machines):
        machine_color = color_map(k_idx)
        current_job = dummy_start

        while current_job != dummy_end:
            next_job = None
            for j in jobs:
                if j != current_job and E[current_job, j, k].X > 0.5:
                    next_job = j
                    break

            if next_job is None:
                break

            chosen_d_in, chosen_d_out = None, None
            for d in depots:
                if v[current_job, k, d].X > 0.5:
                    chosen_d_in = d
                if v[next_job, k, d].X > 0.5:
                    chosen_d_out = d

            if current_job != dummy_end:
                start_time = s[current_job, k, chosen_d_in].X
                finish_time = start_time + P[current_job, chosen_d_in, chosen_d_out]
                ax.barh(
                    y=k_idx, left=start_time, width=finish_time - start_time, height=0.4,
                    color=machine_color, edgecolor='black'
                )
                if current_job not in [dummy_start, dummy_end]:
                    ax.text(
                        (start_time + finish_time) / 2.0, k_idx, f"Job {current_job}",
                        ha='center', va='center', color='black', fontsize=9
                    )

            current_job = next_job

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"Machine {k}" for k in machines])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"Machine Gantt Chart (Makespan = {T_var.X:.2f})")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_job_gantt_solution(solution):
    """
    Plots a job-based Gantt chart:
      - Skips dummy jobs (0 and N+1).
      - Clamps the full_tray_time so it does not exceed the job's process time.
      - Displays:
         1) Release time (blue marker)
         2) A bar from [release_t, tray_t] if tray_t > release_t
         3) Full tray time (green marker)
         4) A wait segment from [tray_t, arrival_t]
         5) A service segment from [arrival_t, arrival_t + job_proc_time]

    Parameters
    ----------
    solution : dict
        A dictionary returned by your scheduling model, containing:
          - "model": Gurobi model
          - "v": v[i,k,d] decision variables
          - "s": s[i,k,d] decision variables (machine start time for job i)
          - "jobs": list of all job indices (0..N+1)
          - "machines": list of machine indices
          - "depots": list of depot indices
          - "release_times": 2D array => release_times[i, d]
          - "full_tray_times": 1D array => full_tray_times[i]
          - "process_times": 3D array => process_times[i, d, k]
            (or however you store process time per job/depot/machine)
    """

    model          = solution["model"]
    v              = solution["v"]
    s              = solution["s"]
    jobs           = solution["jobs"]
    machines       = solution["machines"]
    depots         = solution["depots"]

    # In this example, we assume:
    # release_times[i, d]   -> the release time of job i at depot d
    # full_tray_times[i]    -> the base tray time (clamped if needed)
    # process_times[i, d, k]-> the job i's process time given depot d and machine k

    release_times  = solution["release_times"]    
    full_tray_times = solution["full_tray_times"]
    process_times  = solution["process_times"]    

    # Check feasibility of the model
    if model.SolCount == 0:
        print("No feasible solution found; cannot plot Gantt chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Identify the dummy start/end jobs
    dummy_start = 0
    dummy_end   = len(jobs) - 1

    # A color map for real jobs
    color_map = plt.cm.get_cmap("tab20", len(jobs))

    for i in jobs:
        # Skip dummy jobs
        if i in [dummy_start, dummy_end]:
            continue

        # Find the assigned machine k, depot d (i.e. v[i,k,d] == 1)
        assigned_k = None
        assigned_d = None
        for k in machines:
            for d in depots:
                if v[i, k, d].X > 0.5:
                    assigned_k = k
                    assigned_d = d
                    break
            if assigned_k is not None:
                break

        if assigned_k is None or assigned_d is None:
            # job i not assigned in a feasible manner
            continue

        # (1) Release time for this job i at depot assigned_d
        #     If your indexing is i-1 for real jobs, adapt as needed.
        release_t = release_times[i-1, assigned_d]

        # (2) Show a marker at the release time
        ax.scatter(
            [release_t], [i],
            color='blue', marker='|', s=150,
            label="Release Time" if i == jobs[1] else ""
        )

        # (3) The process time for job i given (d, k)
        job_proc_time = process_times[i, assigned_d, assigned_k]

        # (4) Clamp the full tray time so it does not exceed process time
        #     If your indexing is i-1 for real jobs, adapt as needed.
        raw_tray_time = full_tray_times[i-1]
        tray_t = min(raw_tray_time, job_proc_time)

        # (5) If tray_t > release_t, we can plot a bar from [release_t, tray_t]
        if tray_t > release_t:
            ax.barh(
                y=i,
                left=release_t,
                width=tray_t - release_t,
                height=0.4,
                color='orange',
                edgecolor='black'
            )
            ax.text(
                (release_t + tray_t)/2.0,
                i,
                "R→F",
                ha='center',
                va='center',
                color='black',
                fontsize=9
            )

        # Show a marker at the (possibly clamped) full tray time
        ax.scatter(
            [tray_t], [i],
            color='green', marker='|', s=150,
            label="Full Tray Time" if i == jobs[1] else ""
        )

        # (6) Machine arrival time from the solution
        arrival_t = s[i, assigned_k, assigned_d].X

        # (7) Wait segment: from [tray_t, arrival_t]
        if arrival_t > tray_t:
            ax.barh(
                y=i,
                left=tray_t,
                width=arrival_t - tray_t,
                height=0.4,
                color='lightgray',
                edgecolor='black'
            )
            ax.text(
                (tray_t + arrival_t)/2.0,
                i,
                "WAIT",
                ha='center',
                va='center',
                color='black',
                fontsize=9
            )

        # (8) Service segment: [arrival_t, arrival_t + job_proc_time]
        service_start  = arrival_t
        service_finish = arrival_t + job_proc_time

        ax.barh(
            y=i,
            left=service_start,
            width=job_proc_time,
            height=0.4,
            color=color_map(i),
            edgecolor='black'
        )
        ax.text(
            (service_start + service_finish)/2.0,
            i,
            f"Job {i}\n(M{assigned_k},D{assigned_d})",
            ha='center',
            va='center',
            color='white',
            fontsize=9
        )

    # Final cosmetics
    ax.set_yticks(jobs)
    ax.set_yticklabels([f"Job {j}" for j in jobs])
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Job Index")
    ax.set_title("Job Gantt Chart (Skip Dummy, Box Between Release and Tray)")

    # Only show legend if labels exist
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def visualize_settings_with_full_tray_locations(job_locations, depots, full_tray_times, machine_depots, picker_speed=0.5, 
                                                num_rows=10, row_width=1.5, row_length=25, headland_space=5):
    """
    Visualize the randomized settings with vertical rows, job locations, depots, machines, and full tray locations.
    
    Parameters:
    - job_locations: ndarray, shape (num_jobs, 2), coordinates of jobs.
    - depots: ndarray, shape (2, num_depots), coordinates of depots.
    - full_tray_times: ndarray, release times for each job.
    - machine_depots: ndarray, depot indices where machines are located.
    - picker_speed: float, speed of the picker.
    - num_rows: int, number of rows in the setting.
    - row_width: float, width of each row.
    - row_length: float, length of each row.
    - headland_space: float, vertical space at the bottom of rows.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rows as thick vertical lines
    for row_id in range(num_rows + 1):
        row_x_start = row_id * row_width - row_width / 2
        plt.plot([row_x_start, row_x_start], [headland_space, row_length], color='purple', linewidth=5, alpha=0.5, label='Row' if row_id == 0 else None)
    
    # Plot jobs and label job IDs
    for idx, (x, y) in enumerate(zip(job_locations[:,0], job_locations[:,1])):
        plt.scatter(x, y, color='blue', s=100, label='Jobs' if idx == 0 else None, marker='o')
        plt.text(x, y + 0.5, f"J{idx}", fontsize=10, color='blue', ha='center')  # Label above
    
    # Calculate and plot full tray locations (green circles with golden cross)
    full_tray_ys = job_locations[:,1] - picker_speed * full_tray_times
    for idx, (x, y) in enumerate(zip(job_locations[:,0], full_tray_ys)):
        plt.scatter(x, y, color='green', s=100, label='Full Tray' if idx == 0 else None, marker='o', edgecolors='black', linewidths=1.5)
        plt.text(x, y - 1, f"T{idx}", fontsize=10, color='green', ha='center')  # Label below

    
    # Plot depots and label depot IDs
    for idx, (x, y) in enumerate(zip(depots[0], depots[1])):
        plt.scatter(x, y, color='black', s=100, label='Depots' if idx == 0 else None, marker='s')
        plt.text(x, y + 0.5, f"D{idx}", fontsize=10, color='black', ha='center')  # Label below
    
    # Plot machines near depots (horizontally aligned, with labels below)
    for idx, depot_idx in enumerate(machine_depots):
        depot_x = depots[0][depot_idx]
        depot_y = depots[1][depot_idx]
        machine_x = depot_x + (idx - 1) * 0.5  # Slight horizontal offset
        plt.scatter(machine_x, depot_y, color='red', s=80, label='Machines' if idx == 0 else None, marker='^')
        plt.text(machine_x, depot_y - 1.0, f"M{idx}", fontsize=10, color='red', ha='center')  # Label below
    
    # Formatting
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Visualization of Job, Depot, Machine, and Full Tray Locations")
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_scheduling_model(job_locations, full_tray_times, depot_locations, 
                           machine_availables, machine_depots, picker_speed=0.5):
    # Rt = np.sort(Rt)
    # Rt = np.insert(Rt, 0, 0)  # start job
    # Rt = np.append(Rt, 0)  # end job

    depot_num = len(depot_locations)
    job_num = len(job_locations)
    machine_num = len(machine_availables)

    loading_duration = 5  # loading duration for each job
    unloading_duration = 5  # unloading duration for each job
    
    # list for the modeling
    jobs = list(range(job_num+2))  # 0 for start, N+1 for end, 1..N are real jobs
    depots = list(range(depot_num))  # depots indexed from 0 to D-1
    machines = list(range(machine_num))
    

    # calculate the full tray locations
    full_tray_locations = np.copy(job_locations)
    full_tray_locations[:,1] -= picker_speed*full_tray_times
    full_tray_locations = np.vstack(([0, 0], full_tray_locations, [50, 50]))
    # release_times = np.array([full_tray_times,full_tray_times])
    release_times = np.tile(full_tray_times, (depot_num, 1)).T
    # release_times = np.insert(release_times, 0, 0)  # start job
    # release_times = np.append(release_times, 0)  # end job
    

    for i in range(job_num):
        for d in depots:
            running_time = abs(full_tray_locations[i][0] - depot_locations[d][0]) + abs(full_tray_locations[i][1] - depot_locations[d][1]) 
            release_time = max(full_tray_times[i]-running_time, 0)
            release_times[i,d] = release_time
    Rt = np.vstack((np.ones(depot_num), release_times, np.ones(depot_num)))

    process_times = np.zeros((len(full_tray_locations), depot_num, depot_num))  # Initialize the array with zeros
    A_k = machine_availables.astype(int)
    Rt = Rt.astype(int)
    full_tray_locations = full_tray_locations.astype(int)
    
    D0 = machine_depots
    
    for i in jobs:
        for d1 in depots:
            for d2 in depots:
                if i != 0 and i != job_num + 1:
                    process_times[i][d1][d2] = (
                        abs(full_tray_locations[i][0] - depot_locations[d1][0]) + 
                        abs(full_tray_locations[i][1] - depot_locations[d1][1]) + 
                        loading_duration + 
                        abs(full_tray_locations[i][0] - depot_locations[d2][0]) + 
                        abs(full_tray_locations[i][1] - depot_locations[d2][1]) + 
                        unloading_duration
                    )
                else:
                    process_times[i][d1][d2] = 0
    process_times = process_times.astype(int)
    # print(process_times)
    BigM = 100000  # large M
    model = gp.Model("Strawberry_Pickup")

    # Decision variables
    # v[i,k,d] = 1 if machine k executes job i departing from depot d (for real jobs and also for i=0 or i=N+1)
    v = model.addVars(jobs, machines, depots, vtype=GRB.BINARY, name="v")

    # E[i,j,k] = 1 if machine k executes job i immediately followed by job j
    E = model.addVars(jobs, jobs, machines, vtype=GRB.BINARY, name="E")

    # s[i,k,d]: start time of job i on machine k if departing from depot d
    s = model.addVars(jobs, machines, depots, vtype=GRB.INTEGER, name="s", lb=0.0)

    # Objective: Minimize the maximum completion time
    # Completion time of job i on machine k from depot d is s[i,k,d] + P[i,d,d'] (depending on next job/depot).
    # To handle min max, introduce a variable T representing the makespan and minimize T.
    T = model.addVar(vtype=GRB.INTEGER, name="makespan", lb=0.0)

    model.setObjective(T, GRB.MINIMIZE)

    # Constraints

    # 0) No self loop
    model.addConstrs((E[i,i,k] == 0 for i in jobs for k in machines), name="no_self_loop")

    # 1) Each real job i (1..N) must be done exactly once
    model.addConstrs((gp.quicksum(v[i,k,d] for k in machines for d in depots) == 1 for i in jobs[1:-1]), name=f"job_once_constraints")

    # 2) Each machine starts from D0[k] depot
    model.addConstrs((v[i,k,D0[k]] >= E[0,i,k] for k in machines for i in jobs[1:]), f"start_job_constraints")

    # 3) Each machine starts and ends exactly once at a depot at job 0 and job N+1
    for k in machines:
        # Dummy start jobs. Machine k starts from depot D0[k]
        model.addConstr(v[0,k,D0[k]] == 1, name=f"start_{k}_1")
        model.addConstrs((v[0,k,d] == 0 for d in depots if d != D0[k]), name=f"start_{k}_2")
        # Dummy end jobs
        model.addConstr(gp.quicksum(v[job_num+1,k,d] for d in depots) == 1, name=f"end_{k}")

    # 4) Connectivity constraints:
    # For each machine k and each job i (except end job):
    # Outflow = inflow in terms of E
    for k in machines:
        for i in jobs:
            outflow = gp.quicksum(E[i,j,k] for j in jobs if j != i)
            inflow = gp.quicksum(E[h,i,k] for h in jobs if h != i)
            # If i is start job (0): outflow = v[0,k,d], inflow=0
            # If i is end job (N+1): inflow = v[N+1,k,d], outflow=0
            # If i is a real job: inflow = v[i,k,d], outflow = v[i,k,d]
            
            if i == 0:
                # Start job: outflow = sum_d v[0,k,d], inflow = 0
                model.addConstr(outflow == gp.quicksum(v[0,k,d] for d in depots), f"flow_start_{k}")
                model.addConstr(inflow == 0, f"flow_start_{k}_in")
            elif i == job_num+1:
                # End job: inflow = sum_d v[N+1,k,d], outflow=0
                model.addConstr(inflow == gp.quicksum(v[job_num+1,k,d] for d in depots), f"flow_end_{k}")
                model.addConstr(outflow == 0, f"flow_end_{k}_out")
            else:
                # Real job: inflow = outflow = sum_d v[i,k,d]
                model.addConstr(outflow == gp.quicksum(v[i,k,d] for d in depots), f"outflow_job_{i}_{k}")
                model.addConstr(inflow == gp.quicksum(v[i,k,d] for d in depots), f"inflow_job_{i}_{k}")

    # 5) Timeline constraints:
    # For any pair (i,j) such that E[i,j,k] = 1, we must have:
    # s[j,k,d'] >= s[i,k,d] + P[i,d,d'] - BigM*(1 - E[i,j,k])
    # We need to know which depot is associated with job i and j under machine k
    # This is tricky because v[i,k,d]=1 chooses the depot d for job i.
    # We can enforce this by summation over d and d':
    #   s[j,k,d'] >= s[i,k,d] + P[i,d,d'] - BigM(1 - E[i,j,k])
    # and also link that if v[i,k,d]=1 and v[j,k,d']=1 then these must hold.
    #
    # One way is to enforce for all d,d':
    # s[j,k,d'] >= s[i,k,d] + P[i,d,d'] - BigM*(1 - E[i,j,k]) - BigM*(1 - v[i,k,d]) - BigM*(1 - v[j,k,d'])
    # This is a big constraint. Alternatively, you can linearize depot choice more cleverly.
    #
    # For simplicity, assume you know P[i,d,d'] and apply constraints for all d,d' with big-M:
    for k in machines:
        for i in jobs[1:-1]:
            for j in jobs[1:]:
                if i != j:
                    for d_in in depots:
                        for d_out in depots:
                            model.addConstr(s[j,k,d_out] 
                                            >= s[i,k,d_in] + process_times[i,d_in,d_out]
                                            - BigM*(1 - E[i,j,k]) 
                                            - BigM*(1 - v[i,k,d_in])
                                            - BigM*(1 - v[j,k,d_out]),
                                            name=f"time_seq_{i}_{j}_{k}_{d_in}_{d_out}")

    # 6) Availability constraints:
    # s[i,k,d] >= A_k + t_i * v[i,k,d]
    for k in machines:
        for i in jobs[1:]:
            for d in depots:
                model.addConstr(s[i,k,d] >= A_k[k], f"avail_{i}_{k}_{d}")
                model.addConstr(s[i,k,d] >= Rt[i,d], f"release_{i}_{k}_{d}")

    # 7) Objective linking:
    # T >= s[i,k,d] + P[i,d,d'] * v[i,k,d] for all i,k,d,d'
    for k in machines:
        for i in jobs[1:]:
            for d in depots:
                model.addConstr(T >= s[i,k,d], f"makespan_{i}_{k}_{d}")

    # Tuning gurobi parameters
    model.Params.Heuristics = 0.707 # Set the heuristic parameter to 0.707
    # model.Params.MIPFocus = 2 # Set the MIP focus to 2 for more aggressive cut
    # model.Params.NoRelHeurTime = 60 # Set the time limit for heuristic to 100 seconds
    #model.Params.NoRelHeurWork = 1e12 # Set the work limit for heuristic to 1e6 iterations
    # Solve
    model.optimize()
        
    return {
            "model": model,
            "E": E,
            "v": v,
            "s": s,
            "P": process_times,
            "machines": machines,
            "jobs": jobs,
            "depots": depots,
            "T": T,
            "release_times": release_times,
            "process_times": process_times,
            "full_tray_times": full_tray_times,
            "full_tray_locations": full_tray_locations
        }



if __name__ == "__main__":
    multi_depot_demo(video=False)
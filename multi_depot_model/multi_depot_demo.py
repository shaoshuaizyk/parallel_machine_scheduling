import gurobipy as gp
from gurobipy import GRB
import numpy as np

def multi_depot_demo(length=100, width=100, N=5, D=2, D0=None, M_num=2, A0=None, seed=42, print_sol=True, visualize=False, video=False):
    # Define the region of the strawberry field
    length = length
    width = width

    # Given data 
    N = N # number of jobs
    D = D # number of depots
    M_num = M_num # number of machines

    jobs = list(range(N+2))  # 0 for start, N+1 for end, 1..N are real jobs
    depots = list(range(D))  # depots indexed from 0 to D-1
    machines = list(range(M_num))

    # Generate random call places and correspoding release times
    seed = seed
    rng = np.random.default_rng(seed)
    Rt = rng.integers(0, length*2, N) # release times for jobs (for i=0 and i=N+1 could be 0)
    Rt = np.sort(Rt)
    Rt = np.insert(Rt, 0, 0)  # start job
    Rt = np.append(Rt, 0)  # end job
    A_k = rng.integers(0, length//5, M_num) if A0 is None else np.array(A0) # availability times for machines
    job_xs = rng.integers(0, length, N) # x coordinates of jobs
    job_xs = np.insert(job_xs, 0, 0)  # start job
    job_xs = np.append(job_xs, length)  # end job
    job_ys = rng.integers(0, width, N) # y coordinates of jobs
    job_ys = np.insert(job_ys, 0, 0)  # start job
    job_ys = np.append(job_ys, length)  # end job
    depot_xs = np.array([int(_*length/(D+1)) for _ in range(1, D+1)]) # x coordinates of depots
    depot_ys = np.array([0 for _ in range(D)]) # y coordinates of depots
    D0 = rng.integers(0, D, M_num) if D0 is None else np.array(D0) # Starting depots for machines

    loading_duration = 5  # loading duration for each job
    unloading_duration = 5  # unloading duration for each job
    P = np.array([[[abs(job_xs[i] - depot_xs[d1]) + abs(job_ys[i] - depot_ys[d1]) + loading_duration + abs(job_xs[i] - depot_xs[d2]) + abs(job_ys[i] - depot_ys[d2]) + unloading_duration if i != 0 and i != N+1 else 0 for d2 in range(D) ] for d1 in range(D)] for i in jobs]) # Manhattan distances between depots and jobs

    BigM = N*length*2  # large M

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
        model.addConstr(gp.quicksum(v[N+1,k,d] for d in depots) == 1, name=f"end_{k}")

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
            elif i == N+1:
                # End job: inflow = sum_d v[N+1,k,d], outflow=0
                model.addConstr(inflow == gp.quicksum(v[N+1,k,d] for d in depots), f"flow_end_{k}")
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
                                            >= s[i,k,d_in] + P[i,d_in,d_out]
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
                model.addConstr(s[i,k,d] >= Rt[i], f"release_{i}_{k}_{d}")

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

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        if print_sol:
            print("Optimal solution found with makespan:", T.X)
            print(f"Machine starting depots: {D0}")
            print(f"Release times: {Rt[1:-1]}")
            print(f"Availability times: {A_k}")
            print(f"Job coordinates: {list(zip(job_xs[1:-1], job_ys[1:-1]))}")
            print(f"Depot coordinates: {list(zip(depot_xs, depot_ys))}")
            print(f"Job costs: {[tuple([[P[i, d_in, d_out] for d_out in depots] for d_in in depots]) for i in jobs[1:-1]]}")
            # Print routes and schedules in time order by machines. 
            for k in machines:
                print(f"Machine {k}:")
                i = 0
                while i != N+1:
                    for j in jobs:
                        if E[i,j,k].X > 0.9:
                            if i != 0:
                                start_d = None
                                end_d = None
                                for d in depots:
                                    if v[i,k,d].X > 0.9:
                                        assert start_d is None, "Multiple start depots found"
                                        start_d = d
                                    if v[j,k,d].X > 0.9:
                                        assert end_d is None, "Multiple end depots found"
                                        end_d = d
                                print(f"Job {i} at {s[i,k,start_d].X}s in depot {start_d} -> Job {j} at {s[i,k,start_d].X + P[i,start_d,end_d]}s in depot {end_d}")
                            i = j
                            break
                print()
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,10))
            for d in depots:
                plt.scatter(depot_xs[d], depot_ys[d], color='black', s=100, marker='s', label=f"Depot {d}")
            for i in jobs[1:-1]:
                plt.scatter(job_xs[i], job_ys[i], color='blue', s=50, marker='o', label=f"Job {i}")
            for k in machines:
                for i in jobs[1:-1]:
                    for d in depots:
                        if v[i,k,d].X > 0.9:
                            plt.plot([depot_xs[d], job_xs[i]], [depot_ys[d], job_ys[i]], color='red')
            plt.show()
        if video: # TODO: Generate a video showing the process of machine moving from the start depot to the job place at Manhattan distance, loading the fruits, and moving back to the end depot. Show the job position when it is released, remove the job after it is loading on the machine. Show the previous a few seconds history trajectory of the machine. Save the video as mp4.
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_xlim(0, length)
            ax.set_ylim(0, width)
            for d in depots:
                ax.scatter(depot_xs[d], depot_ys[d], color='black', s=100, marker='s', label=f"Depot {d}")
            for i in jobs[1:-1]:
                ax.scatter(job_xs[i], job_ys[i], color='blue', s=50, marker='o', label=f"Job {i}")
            lines = []
            for k in machines:
                for i in jobs[1:-1]:
                    for d in depots:
                        if v[i,k,d].X > 0.9:
                            line, = ax.plot([depot_xs[d], job_xs[i]], [depot_ys[d], job_ys[i]], color='red')
                            lines.append(line)
            def update(frame):
                for line in lines:
                    line.set_data([depot_xs[d], job_xs[i]], [depot_ys[d], job_ys[i]])
            ani = animation.FuncAnimation(fig, update, frames=100, blit=True)
            ani.save('strawberry_pickup.mp4', writer='ffmpeg', fps=10)

    
    else:
        raise Exception("No solution found")

if __name__ == "__main__":
    multi_depot_demo()
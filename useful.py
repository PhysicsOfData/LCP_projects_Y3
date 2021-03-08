import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
from time import time as read_time
from scipy.stats import kurtosis, skew 
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import widgets, Layout
from IPython.display import display, display_html, HTML
from matplotlib.ticker import FormatStrFormatter


###	INPUT: n° rows and cols of the subplots grid
###			the kwargs that must be forwarded to plt.figure
###	OUTPUT: the figure, the list of its axis
### AIM: create an interface equal to plt.subplots() but for 3D plots
def subplots3d(rows=1, cols=1, **figkwargs):
	fig = plt.figure(**figkwargs)
	idx = 1
	for i in range(rows):
		for j in range(cols):
			fig.add_subplot(rows, cols, idx, projection="3d")
			idx = idx + 1
	return fig, fig.get_axes()


### INPUT: the string with the path to the snapshot file to be loaded
### OUTPUT: the tuple (t, L, data), where:
###			"t" is the time of the snapshot
###			"L" is the list of the labels corresponding to the columns of the numpy array
###			"data" is the numpy array containing the quantities in the snapshot
### AIM: it is used by "simulation_to_pandas"
###			can be used to load single files when needed
def loadfile(filename):
	with open(filename, "r") as file:
		first_line = file.readline()
		time = float(first_line.split()[-1])
		particles_data = np.loadtxt(file)
		labels = ["x", "y", "z", "vx", "vy", "vz", "m", "idx"]
	return time, labels, particles_data


### INPUT: the list of strings of paths to snapshot files to be inserted in the same DataFrame
### OUTPUT: the DataFrame of the simulation in the form:
###						x	y	z	vx	vy	vz	m	idx	t
###			t1	idx1
###				idx2
###				...
###			t2	idx1
###				idx2
###				...
###			...
### AIM: it is used to load several files, like the whole simulation
###			it can be used also to load only some files
def simulation_to_pandas(filelist):
	labels = ["x", "y", "z", "vx", "vy", "vz", "m", "idx", "t"]
	types = [float, float, float, float, float, float, float, int, int]
	dataset = pd.DataFrame(columns=labels)
	print("Loading database")
	for i,filename in enumerate(filelist):
		print(str(100*i//len(filelist))+"%", end="\r", flush=True)
		t, _, data = loadfile(filename)
		t = np.full(data.shape[0], t).astype(int).reshape((data.shape[0], 1))
		values = np.hstack((data, t))
		dataset = dataset.append(pd.DataFrame(values, columns=labels))
	print("Loading complete\n")
	dataset.set_index(["t", "idx"], drop=False, inplace=True)
	return dataset


### INPUT: the whole DataFrame and the number of random particles wanted in the sliced DataFrame
### OUTPUT: the sliced DataFrame
### AIM: after a simulation is loaded, take a slice of "N" random particles at every time
def take_random_particles(simulation, N):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# ensure to take only surviving particles
	final_time = subdata["t"].to_numpy().max()
	survived_particles = subdata.loc[final_time]["idx"].to_numpy()
	# select N at random
	selected_particles = np.sort(np.random.choice(survived_particles, size=N, replace=False))
	subdata = subdata.loc(axis=0)[pd.IndexSlice[:,selected_particles]]
	# clean unused memory
	gc.collect()
	return subdata
### Inverse version
def drop_random_particles(simulation, N):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# ensure to drop among all the particles
	initial_time = subdata["t"].to_numpy().min()
	all_particles = subdata.loc[initial_time]["idx"].to_numpy()
	# drop N at random
	dropping_particles = np.sort(np.random.choice(all_particles, size=N, replace=False))
	subdata.drop(index=dropping_particles, level="idx", inplace=True)
	# clean unused memory
	gc.collect()
	return subdata


### INPUT: the whole DataFrame and the list of particles wanted in the sliced DataFrame
### OUTPUT: the sliced DataFrame
### AIM: after a simulation is loaded, take a slice of some specified particles at every time
def take_these_particles(simulation, particles_idx):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# select the particles
	subdata = subdata.loc(axis=0)[pd.IndexSlice[:,particles_idx]]
	# clean unused memory
	gc.collect()
	return subdata
### Inverse version
def drop_these_particles(simulation, particles_idx):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# drop the particles
	subdata.drop(index=particles_idx, level="idx", inplace=True)
	return subdata


### INPUT: the whole DataFrame and the number of time-steps wanted in the sliced DataFrame
### OUTPUT: the sliced DataFrame
### AIM: after a simulation is loaded, take a slice of "N" random time-steps for every particle
def take_random_times(simulation, N):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# select the times
	all_times = np.unique(simulation["t"].to_numpy())
	selected_times = np.sort(np.random.choice(all_times, size=N, replace=False))
	# reduce the dataset
	subdata = simulation.loc[selected_times]
	# restore the dataset
	subdata.set_index(["t", "idx"], drop=False, inplace=True)
	# clean unused memory
	gc.collect()
	return subdata
### Inverse version
def drop_random_times(simulation, N):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# drop the times
	all_times = np.unique(simulation["t"].to_numpy())
	dropping_times = np.sort(np.random.choice(all_times, size=N, replace=False))
	subdata.drop(index=dropping_times, level="t", inplace=True)
	# clean unused memory
	gc.collect()
	return subdata


### INPUT: the whole DataFrame and the time-steps wanted in the sliced DataFrame
### OUTPUT: the sliced DataFrame
### AIM: after a simulation is loaded, take a slice of some specified time-steps for every particle
def take_these_times(simulation, times):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# reduce the dataset
	subdata = subdata.loc[times]
	# restore the dataset
	subdata.set_index(["t", "idx"], drop=False, inplace=True)
	# clean unused memory
	gc.collect()
	return subdata
### Inverse version
def drop_these_times(simulation, times):
	# prepare the sub-dataset
	subdata = simulation.set_index(["t", "idx"], drop=False)
	# drop the times
	subdata.drop(index=times, level="t", inplace=True)
	# clean unused memory
	gc.collect()
	return subdata


###	INPUT: the DataFrame to extract times from
###	OUTPUT: the numpy array of the times
### AIM: method to simplify the syntax
def get_times(simulation):
	return np.unique(simulation["t"].to_numpy())
###	INPUT: the DataFrame to extract particle indices from
###	OUTPUT: the numpy array of the particle indices
### AIM: method to simplify the syntax
def get_idx(simulation):
	return np.unique(simulation["idx"].to_numpy())


###	INPUT: the snapshot to use to calculate the COM
###	OUTPUT: the pd.DataFrame containing the position of the COM
###	AIM: to be used alone or in other functions
def centre_of_mass(snapshot):
	array = snapshot.to_numpy()
	masses = array[:,6]    
	coordx = array[:,0] * masses
	coordy = array[:,1] * masses
	coordz = array[:,2] * masses
	# dividing into + terms and - terms
	xpos, xneg = coordx[coordx >= 0], coordx[coordx < 0]
	ypos, yneg = coordy[coordy >= 0], coordy[coordy < 0]
	zpos, zneg = coordz[coordz >= 0], coordz[coordz < 0]
	# sorting to have smaller terms at the beginning of the sum
	def absolute_sorting(array):
		order = np.argsort(np.abs(array))
		return array[order]
	masses = np.sort(masses)
	xpos, xneg = absolute_sorting(xpos), absolute_sorting(xneg)
	ypos, yneg = absolute_sorting(ypos), absolute_sorting(yneg)
	zpos, zneg = absolute_sorting(zpos), absolute_sorting(zneg)
	# totals
	tot_mass = np.sum(masses)
	x_CM = (np.sum(xpos) + np.sum(xneg)) / tot_mass
	y_CM = (np.sum(ypos) + np.sum(yneg)) / tot_mass
	z_CM = (np.sum(zpos) + np.sum(zneg)) / tot_mass
	# position of COM
	COM = np.array([[x_CM, y_CM, z_CM]])
	return pd.DataFrame(COM, index=pd.Index(get_times(snapshot), name="t"), columns=["x", "y", "z"])


###	INPUT: the snapshot and the COM to compute the distance between
###	OUTPUT: the numpy array of the distances
###	AIM: compute the distance between some particles and the COM
def distance_from_COM(snapshot, COM_pos):
	array = snapshot.to_numpy()[:,0:3]
	COM_pos = COM_pos.to_numpy().flatten()
	COM_pos = np.array((COM_pos,) * array.shape[0])
	distances = array - COM_pos
	return np.linalg.norm(distances, axis=1)


###	INPUT:
###	OUTPUT:
###	AIM:
def smallest_values(distances, threshold=0.9):
	distances = distances
	ordered_distances = np.sort(distances)
	max_dist = ordered_distances[int(threshold * distances.size)]
	filtered_distances = np.where(ordered_distances <= max_dist, ordered_distances, -1)
	to_keep = np.isin(distances, filtered_distances)
	return to_keep


###	INPUT: the snapshot to compute the COM on
###			the accuracy wanted for the position of the COM
###	OUTPUT:	the pd.Series of the positions of the COM
###			the sub-snapshot on which the last COM is computed
###	AIM: this function is for showing consistency of this method
###			the funcion COM_over_core must be preferred for the results
def iterative_COM_core(snapshot, threshold=0.9, accuracy=1e-10, N=50):
	COMs = np.full((1, 3), centre_of_mass(snapshot).to_numpy())
	for i in range(1, N):
		distances = distance_from_COM(snapshot, pd.DataFrame(COMs[i-1]))
		mask = smallest_values(distances, threshold)
		COMs = np.vstack((COMs, centre_of_mass(snapshot[mask]).to_numpy()))
		if np.linalg.norm(COMs[i] - COMs[i-1]) < accuracy:
			break
	return COMs, snapshot[mask]


###	INPUT: same arguments to be forwarded to iterative_COM_core
###	OUTPUT: the final (best) COM pos found by iterative_COM_core
###	AIM: to simplify the usage of the code
def COM_over_core(*args):
	COM, _ = iterative_COM_core(*args)
	return COM[-1,:]


###	INPUT: the simulation and the number of most massive particles
###	OUTPUT: the dataframe of the N most massive particles, with index "idx" and column "m"
###	AIM: simplify the function call
def find_largest_mass(simulation, order=1):
    t = get_times(simulation).max()
    subdata = simulation.loc[t][["m"]]
    subdata.drop_duplicates(inplace=True)
    subdata.sort_values("m", ascending=False, inplace=True)
    return take_these_particles(simulation, subdata.index[order-1])


###	INPUT: the subdataset (can include more than one particle) and the title
###	OUTPUT: None
###	AIM: simplify the plotting of a 3D trajectory
def path_plot3D(subdata, ax, title=""):
	particles = subdata["idx"].unique()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_title(title)
	trajectory = None
	for particle in particles:
		trajectory = subdata.loc(axis=0)[:,particle]
		ax.plot(trajectory['x'], trajectory['y'], trajectory['z'], linestyle=':', label="Particle "+str(int(particle)))
	ax.legend()
	del trajectory
	gc.collect()
	return


###	INPUT: the whole simulation
###	OUTPUT:	the numpy array of the indices of the vanished particles
###	AIM: simplify the syntax
def get_vanishing_particles(whole_simulation):
	all_particles = pd.unique(whole_simulation["idx"])
	survived_particles = whole_simulation.loc[whole_simulation["t"].max(), "idx"].to_numpy()
	vanished_particles = all_particles[np.isin(all_particles, survived_particles, invert=True)].astype(np.int32)
	del all_particles, survived_particles
	gc.collect()
	return vanished_particles


###	INPUT: simulation or subdata
###	OUTPUT:	boolean ensuring that every mass is constant along all the timestamps
###	AIM: just show what we have proven
def check_for_merges(simulation):
	constant_mass = np.empty(simulation["idx"].unique().size)
	for idx,trajectory in simulation.groupby(level=1):
		constant_mass[idx] = 1 == trajectory["m"].unique().size
	every_mass_constant = np.all(constant_mass)
	return not every_mass_constant


###	INPUT: the snapshot and the COM to compute the distance between
###	OUTPUT: the numpy array of the distances
###	AIM: compute the distance between some particles and the COM
def relative_position(snapshot, COM_pos):
	array = snapshot.to_numpy()[:,0:3]
	COM_pos = COM_pos.flatten()
	COM_pos = np.array((COM_pos,) * array.shape[0])
	return pd.DataFrame(array - COM_pos, index=snapshot.index, columns=["Dx", "Dy", "Dz"])


def display_grids(df1,df2,caption1="First simulation",caption2="Second simulation"):
	# create output widgets
	widget1 = widgets.Output()
	widget2 = widgets.Output()
	# render in output widgets
	with widget1:
		display(df1.style.set_caption(caption1))
		#df1.info()
	with widget2:
		display(df2.style.set_caption(caption2))
		#df1.info()
	# add some CSS styles to distribute free space
	box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around', width='auto')
	# create Horizontal Box container
	hbox = widgets.HBox([widget1, widget2], layout=box_layout)
	# render hbox
	display_html(hbox)
	return


def find_missing_time(dataset,timestamps):
	groupby_index = dataset[["t"]].groupby("idx")                          # extract list of times associated to each particle
	group_operation = lambda group : group.max()-timestamps[len(group)-1]  # difference between timestamps value and expected time given the number of stamps
	missing_time = groupby_index.agg(group_operation)                      # agg applies function to each group 
	missing_time = missing_time[missing_time["t"]!=0]                      # filters for existing discrepancies
	return missing_time.rename(columns={"t":"missing_time"})


def COM_pos_vel(snapshot):
	array = snapshot.to_numpy()
	masses = array[:,6]
	array = (array[:,0:6] * masses[:,None])
	COM = np.sum(array, axis=0) / np.sum(masses)
	return pd.DataFrame([COM], index=pd.Index(get_times(snapshot), name="t"), columns=["x", "y", "z", "vx", "vy", "vz"])


def compute_COM(simulation, threshold=0.9, accuracy=1e-10, N=50):
	times = get_times(simulation)
	COM = pd.DataFrame(index=pd.Index(times, name="t"), columns=["x", "y", "z", "vx", "vy", "vz"])
	for i,t in enumerate(times):
		print("Processing: "+str(100*i//times.size+1)+"%", end="\r", flush=True)
		snapshot = simulation.loc[t]
		former_COM = COM_pos_vel(snapshot)
		for j in range(1, N):
			distances = distance_from_COM(snapshot, former_COM[["x","y","z"]])
			mask = smallest_values(distances, threshold)
			new_COM = COM_pos_vel(snapshot[mask])
			if np.linalg.norm(new_COM - former_COM) < accuracy:
				break
			former_COM = new_COM
		COM.loc[t] = new_COM.loc[t]
	return COM


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str+=df.to_html(max_rows=10)
    display_html(html_str.replace('table','table style="display:inline"'), raw=True)


def trajectory(dataset,Idx,labels=True):
    if (labels==True) :
        return dataset.loc(axis=0)[pd.IndexSlice[:,Idx]]
    else :
        return dataset.loc(axis=0)[pd.IndexSlice[:,Idx]][labels]



def plot_time_differences(axsa,axsb,dataset,timestamps,problematic_particles,increasing_m_times,increasing_n_times):
    for i in problematic_particles:
        dataset_time_sequence = dataset["t"].loc(axis=0)[pd.IndexSlice[:,i]].to_numpy()  # takes time sequence of anomalous particles
        standard_time_sequence = timestamps[:len(dataset_time_sequence)]
        Nstamps = len(dataset_time_sequence)
        axsa.plot(timestamps[:Nstamps],dataset_time_sequence) # plots dataset time sequence over standard time sequence
        axsb.plot(timestamps[:Nstamps],dataset_time_sequence-timestamps[:Nstamps])  # plots difference of dataset time sequence from standard time sequence 
    axsa.plot(timestamps[:-1],timestamps[1:],linestyle="--",color="turquoise") # off by one line
    axsb.plot(timestamps[:-1],timestamps[1:]-timestamps[:-1],linestyle="--",color="turquoise") # off by one line (meaningful part)
    axsa.plot(increasing_m_times,np.zeros_like(increasing_m_times),"b*",markersize=8) # plotting the points with anomalous mass/number behaviour
    axsa.plot(increasing_n_times,np.zeros_like(increasing_n_times),"b*",markersize=8)
    axsb.plot(increasing_m_times,np.zeros_like(increasing_m_times),"b*",markersize=8) # plotting the points with anomalous mass/number behaviour
    axsb.plot(increasing_n_times,np.zeros_like(increasing_n_times),"b*",markersize=8)
    axsa.grid(True)
    axsb.grid(True)
    axsa.set_xlabel("standard time prograssion")
    axsb.set_xlabel("standard time prograssion")
    axsa.set_ylabel("time")
    axsb.set_ylabel("time difference")
    axsa.legend(list(problematic_particles)+["'one missing' line","mass or number increase"])
    axsb.legend(list(problematic_particles)+["'one missing' line","mass or number increase"])
    return None


def compute_missing_times(dataset,simulation_time_sequence,problematic_particles,increasing_m_times,increasing_n_times):
    missing_time_stamps = []
    for i in problematic_particles:
        particle_time_sequence = dataset["t"].loc(axis=0)[pd.IndexSlice[:,i]].to_numpy()  # takes time sequence of anomalous particles
        Nstamps = len(particle_time_sequence)                                            # number of time frames existing for a givenn particle
        # extracting missing times
        mask = np.convolve( simulation_time_sequence[1:Nstamps+1]-particle_time_sequence ,np.array([-1,1]) ,mode="same") > 0  # registers the position at which the two series collapse into one
        missing_time_stamps = missing_time_stamps + list(simulation_time_sequence[:Nstamps][mask])                  # extracts time value of missing time stamp
    return missing_time_stamps


def missing_frames(dataset,problematic_particles,missing_times):
    T_change = 1900 # time at which the sampling frequency changes
    dt = [ 10 if (t<T_change) else 20 for t in  missing_times]  # distance in time of messing frame from neigbouring frames
    # multi indexes of relevant frames
    previous_idxs = [ (t-ddt,particle) for t,particle,ddt in zip(missing_times,problematic_particles,dt)] 
    new_idxs      = [ (t    ,particle) for t,particle    in zip(missing_times,problematic_particles)   ]
    next_idxs     = [ (t+ddt,particle) for t,particle,ddt in zip(missing_times,problematic_particles,dt)]
    # make new frames by interpolation
    Previous = dataset.loc[pd.IndexSlice[previous_idxs]].assign(t=missing_times).set_index(["t", "idx"],drop=False) # extract and set proper indexes
    Next     = dataset.loc[pd.IndexSlice[next_idxs]].assign(t=missing_times).set_index(["t", "idx"],drop=False)
    New      = ( Previous + Next )/2  # linear interpolation between previous and next steps
    del Previous, Next, previous_idxs, new_idxs, next_idxs, T_change, dt
    return New


def change_to_COM_frame(simulation, COM):
    gb = simulation[["x" ,"y" ,"z" ,"vx" ,"vy" ,"vz"]].groupby("idx")
    t0 = read_time()
    def group_func(x):
        if(len(x.shape)==1):
            return x-COM[x.name].iloc[:x.shape[0]]
        else:
            print(f"{x.name/320:.2f}%\t {(read_time()-t0):.0f} s", end="\r", flush=True)
    relative_1 = gb.transform(group_func)
    relative_1[["m", "idx", "t"]] = simulation[["m", "idx", "t"]]
    relative_1['dist'] = ((relative_1[['x','y','z']]**2).sum(axis=1)).transform(np.sqrt)
    return relative_1


def density_profiling(distance, quantity):
    rang = [ [0,distance.max()], [quantity.min(),quantity.max()]  ]  # range of the binning
    counts, d, q = np.histogram2d(x=distance, y=quantity, bins=[50000,500] , range=rang )
    D = 0.5*(d[1:]+d[:-1])  # create distance mid values
    Q = 0.5*(q[1:]+q[:-1])  # create quantity mid values
    quantity_profile = np.dot( counts, Q)                             			# contraction of quantity values and countings for each value
    error_quantity   = 1/(np.sqrt(counts.sum(axis=1))+1e-15)*quantity_profile 	# rough estimation of error over kinetic energy (from poisson error)
    volume_element   = 4/3*np.pi*(d[1:]**3-d[:-1]**3)                  			# difference of outer sphere and inner shpere produces shell volume
    density_profile  = quantity_profile/volume_element                 			# division of quantity for the corresponding volume
    error_density    = error_quantity/volume_element                   			# propagates error
    return D, density_profile, error_density
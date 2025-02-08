from scipy import interpolate
from scipy import optimize
from scipy import spatial
import numpy as np
import math
import trajectory_planning_helpers as tph
import sys
sys.path.append('../..')
import helper_funcs_glob
import matplotlib.pyplot as plt


# 平滑散点+重采样
# 输入： 笛卡尔坐标系的散点序列
# 输出： 笛卡尔坐标系的重采样散点序列

# ----------------------------------------------------------------------------------------------------------------------
# DISTANCE CALCULATION FOR OPTIMIZATION --------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# return distance from point p to a point on the spline at spline parameter t_glob
def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path)
    s_list = []
    for i in s:
        s_list.append(i.tolist()[0])
    return spatial.distance.euclidean(p, s_list)

debug = True

imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
            "set_new_start": False,                 # set new starting point (changes order, not coordinates)
            "new_start": np.array([0.0, -47.0]),    # [x_m, y_m], set new starting point
            "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
            "num_laps": 1}   


track = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                            file_path='../../inputs/tracks/shanghai.csv',
                                                            width_veh=2.0)

# ------------------------------------------------------------------------------------------------------------------
# LINEAR INTERPOLATION BEFORE SMOOTHING ----------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

track_interp = tph.interp_track.interp_track(track=track,
                                            stepsize=1.0)
track_interp_cl = np.vstack((track_interp, track_interp[0]))

# ------------------------------------------------------------------------------------------------------------------
# SPLINE APPROXIMATION / PATH SMOOTHING ----------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# create closed track (original track)
track_cl = np.vstack((track, track[0]))
no_points_track_cl = track_cl.shape[0]
el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
dists_cum_cl = np.cumsum(el_lengths_cl)
dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)

# find B spline representation of the inserted path and smooth it in this process
# (tck_cl: tuple (vector of knots, the B-spline coefficients, and the degree of the spline))
# k 样条曲线的阶数
# s 样条曲线的平滑度因子
# per 是否闭合
tck_cl, t_glob_cl = interpolate.splprep([track_interp_cl[:, 0], track_interp_cl[:, 1]],
                                    k = 3,
                                    s = 10,
                                    per = 1)[:2]

# calculate total length of smooth approximating spline based on euclidian distance with points at every 0.25m
no_points_lencalc_cl = math.ceil(dists_cum_cl[-1]) * 4
path_smoothed_tmp = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_lencalc_cl), tck_cl)).T
len_path_smoothed_tmp = np.sum(np.sqrt(np.sum(np.power(np.diff(path_smoothed_tmp, axis=0), 2), axis=1)))

# get smoothed path
stepsize_reg = 3.0
no_points_reg_cl = math.ceil(len_path_smoothed_tmp / stepsize_reg) + 1
path_smoothed = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_reg_cl), tck_cl)).T[:-1]

# ------------------------------------------------------------------------------------------------------------------
# PROCESS TRACK WIDTHS ---------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# find the closest points on the B spline to input points
dists_cl = np.zeros(no_points_track_cl)                 # contains (min) distances between input points and spline
closest_point_cl = np.zeros((no_points_track_cl, 2))    # contains the closest points on the spline
closest_t_glob_cl = np.zeros(no_points_track_cl)        # containts the t_glob values for closest points
t_glob_guess_cl = dists_cum_cl / dists_cum_cl[-1]       # start guess for the minimization

for i in range(no_points_track_cl):
# get t_glob value for the point on the B spline with a minimum distance to the input points
    print(t_glob_guess_cl[i])
    print(track_cl[i, :2])

    closest_t_glob_cl[i] = optimize.fmin(dist_to_p,
                                            x0=t_glob_guess_cl[i],
                                            args=(tck_cl, track_cl[i, :2]),
                                            disp=False)

    # evaluate B spline on the basis of t_glob to obtain the closest point
    closest_point_cl[i] = interpolate.splev(closest_t_glob_cl[i], tck_cl)

    # save distance from closest point to input point
    dists_cl[i] = math.sqrt(math.pow(closest_point_cl[i, 0] - track_cl[i, 0], 2)
                            + math.pow(closest_point_cl[i, 1] - track_cl[i, 1], 2))

if debug:
    print("Spline approximation: mean deviation %.2fm, maximum deviation %.2fm"
            % (float(np.mean(dists_cl)), float(np.amax(np.abs(dists_cl)))))

    # get side of smoothed track compared to the inserted track
    sides = np.zeros(no_points_track_cl - 1)

for i in range(no_points_track_cl - 1):
    sides[i] = tph.side_of_line.side_of_line(a=track_cl[i, :2],
                                                b=track_cl[i+1, :2],
                                                z=closest_point_cl[i])

    sides_cl = np.hstack((sides, sides[0]))

    # calculate new track widths on the basis of the new reference line, but not interpolated to new stepsize yet
    w_tr_right_new_cl = track_cl[:, 2] + sides_cl * dists_cl
    w_tr_left_new_cl = track_cl[:, 3] - sides_cl * dists_cl

    # interpolate track widths after smoothing (linear)
    w_tr_right_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_right_new_cl)
    w_tr_left_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_left_new_cl)

    track_reg = np.column_stack((path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]))


track_reg_cl = np.vstack((track_reg, track_reg[0]))
el_lengths_reg_cl = np.sqrt(np.sum(np.power(np.diff(track_reg_cl[:, :2], axis=0), 2), axis=1))
dists_smt_cum_cl = np.cumsum(el_lengths_reg_cl)


# Result Review
# [Length]
print('original track size is ' + str(np.size(track,0)))
print('processed track size is ' + str(np.size(track_reg,0)))
# [Position]
plt.figure(1)
plt.plot(track[:,0], track[:,1],color='red')
plt.plot(track_reg[:,0], track_reg[:,1], color = 'g')
plt.plot(track[:,0], track[:,1],'s',color='red')
plt.plot(track_reg[:,0], track_reg[:,1],'x',color='g')
plt.legend(['Original','Smoothed'])
plt.axis('equal')

# Curvature
kappa_org = []
for i in range(len(track)-2):
    org_x  =  track[i:i+3,0]
    org_y  =  track[i:i+3,1]
    kappa_org.append(tph.cal_curv_points.cal_curv_points(org_x, org_y))
kappa_smt = []
for i in range(len(track_reg)-2):
    smt_x  =  track_reg[i:i+3,0]
    smt_y  =  track_reg[i:i+3,1]
    kappa_smt.append(tph.cal_curv_points.cal_curv_points(smt_x, smt_y))
    
plt.figure(2)    
plt.plot(dists_cum_cl[:-3],kappa_org)
plt.plot(dists_smt_cum_cl[:-2], kappa_smt)
plt.legend(['Original Kappa','Smoothed Kappa'])
plt.show()


"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,kpt):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model

    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.kpt = kpt
  def update(self,bbox, kpt):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
    self.kpt = kpt
  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

#------------------------------------------------------------------------------
# ở một frame thì sẽ có các bbox của detect, track

def cal_pose_simi(kpt1, kpt2):
    coors1, coors2, confs1, confs2 = [], [], [], []
    for i in range(len(kpt1)):
        if i%3 == 2:
            confs1.append(kpt1[i])
            confs2.append(kpt2[i])
        else:
            coors1.append(kpt1[i])
            coors2.append(kpt2[i])
    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    # res = cos(torch.Tensor(coors1), torch.Tensor(coors2))
    # return res.item()
    #-----------------------------------------------
    #x0: 6, 7, 8: x, y, conf i*3+6, i*3+
    diff_hand_right = cosine(coors1[14]-coors1[10], coors1[15]-coors1[11], coors2[14]-coors2[10], coors2[15]-coors2[11]) + \
                      cosine(coors1[18]-coors1[14], coors1[19]-coors1[15], coors2[18]-coors2[14], coors2[19]-coors2[15])
    diff_hand_left = cosine(coors1[16]-coors1[12], coors1[17]-coors1[13], coors2[16]-coors2[12], coors2[17]-coors2[13]) + \
                      cosine(coors1[20]-coors1[16], coors1[21]-coors1[17], coors2[20]-coors2[16], coors2[21]-coors2[17])
    diff_head_right = cosine(coors1[12]-coors1[8], coors1[13]-coors1[9], coors2[12]-coors2[8], coors2[13]-coors2[9])
    diff_head_left = cosine(coors1[10]-coors1[6], coors1[11]-coors1[7], coors2[10]-coors2[6], coors2[11]-coors2[7])
    diff_chest = cosine(coors1[12]-coors1[10], coors1[13]-coors1[11], coors2[12]-coors2[10], coors2[13]-coors2[11]) + \
                  cosine(coors1[12]-coors1[24], coors1[13]-coors1[25], coors2[12]-coors2[24], coors2[13]-coors2[25]) + \
                  cosine(coors1[10]-coors1[22], coors1[11]-coors1[23], coors2[10]-coors2[22], coors2[11]-coors2[23]) + \
                  cosine(coors1[22]-coors1[24], coors1[23]-coors1[25], coors2[22]-coors2[24], coors2[23]-coors2[25])
    diff_leg = cosine(coors1[24]-coors1[28], coors1[25]-coors1[29], coors2[24]-coors2[28], coors2[25]-coors2[29]) + \
                cosine(coors1[28]-coors1[32], coors1[29]-coors1[33], coors2[28]-coors2[32], coors2[29]-coors2[33]) + \
                cosine(coors1[22]-coors1[26], coors1[23]-coors1[27], coors2[22]-coors2[26], coors2[23]-coors2[27]) + \
                cosine(coors1[26]-coors1[30], coors1[27]-coors1[31], coors2[26]-coors2[30], coors2[27]-coors2[31])
    return diff_hand_right + diff_hand_left + diff_head_right + diff_head_left + diff_chest + diff_leg
    
def cosine(x1, y1, x2, y2):
    return abs((x1*y1+x2*y2)/np.sqrt(x1*x1+y1*y1)/np.sqrt(x2*x2+y2*y2))

def cal_pose_simi1(kpt1, kpt2):
    keypoints = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])
    vars = (keypoints * 2) ** 2
    area = 1
    summation = 0
    k1 = 0
    # xd = kpt1[0]
    # yd = kpt1[1]
    # vd = kpt1[2]
    # xg = kpt2[0]
    # yg = kpt2[1]
    # vg = kpt2[2]
    coors1, coors2, confs1, confs2 = [], [], [], []
    for i in range(len(kpt1)):
        if i%3 == 2:
            confs1.append(kpt1[i])
            confs2.append(kpt2[i])
        else:
            coors1.append(kpt1[i])
            coors2.append(kpt2[i])
            if (kpt2[i] > 0):
              k1 += 1
    for i in range(0, len(coors1), 2):
        xd = coors1[i]
        yd = coors1[i+1]
        xg = coors2[i]
        yg = coors2[i+1]
        dx = (xd - xg) * (confs2[i//2] > 0.2)
        dy = (yd - yg) * (confs2[i//2] > 0.2)
        e = (dx ** 2 + dy ** 2) / (vars[i//2]*2)
        summation += np.exp(-e)
    #print('vg = ', kpt2[:])
    #print(f'xd = {xd}, yd = {yd}, xg = {xg}, yg = {yg}, bbox0 = {bbox2[0]}, bbox1 = {bbox2[1]}, bbox2 = {bbox2[2]}, bbox3 = {bbox2[3]}')
    #k1 = np.count_nonzero(kpt2 > 0.2)
    #e = (dx**2 + dy**2) /(vars*(area + np.spacing(1)) *2)
    result = summation / k1
    return result

def kpts_call(kpts, kpts_prev):
  kpt_matrix = np.zeros((len(kpts), len(kpts_prev)))
  # for i in range(num_people):
  #   kpt_matrix.append([])
  #   for j in range(num_people):
  #     kpt_matrix[i].append(0)
  minn = 999
  maxx = 0
  if len(kpts) == 0:
    return np.empty((0,2), dtype=int), np.arange(len(kpts)), np.empty((0, 5), dtype=int)
  if len(kpts) == 1 and len(kpts_prev) == 1:
    return np.array([[1]], dtype = np.float64)
  for i in range(len(kpts)):
    kpt = kpts[i]
    for j in range(len(kpts_prev)):
      kpt_prev = kpts_prev[j]
      kpt_matrix[i][j] = cal_pose_simi(kpt, kpt_prev)
      if kpt_matrix[i][j] > maxx:
        maxx = kpt_matrix[i][j]
      if kpt_matrix[i][j] < minn:
        minn = kpt_matrix[i][j]
  for i in range(len(kpts)):
    for j in range(len(kpts_prev)):
      kpt_matrix[i][j] = (kpt_matrix[i][j] - minn ) / (maxx - minn)
  return kpt_matrix

def kpts_call1(detections, trackers):
  kpts = detections[:, 6:]
  kpts_prev = trackers[:, 6:]
  kpt_matrix = np.zeros((len(kpts), len(kpts_prev)))
  # for i in range(num_people):
  #   kpt_matrix.append([])
  #   for j in range(num_people):
  #     kpt_matrix[i].append(0)
  if len(kpts) == 0:
    return np.empty((0,2), dtype=int), np.arange(len(kpts)), np.empty((0, 5), dtype=int)
  for i in range(len(kpts)):
    kpt = kpts[i]
    for j in range(len(kpts_prev)):
      kpt_prev = kpts_prev[j]
      kpt_matrix[i][j] = cal_pose_simi1(kpt, kpt_prev)
  return kpt_matrix

def cal_pose_simi2(kpt1, kpt2):
    
    coors1, coors2, confs1, confs2 = [], [], [], []
    for i in range(len(kpt1)):
        if i%3 == 2:
            confs1.append(kpt1[i])
            confs2.append(kpt2[i])
        else:
            coors1.append(kpt1[i])
            coors2.append(kpt2[i])
    foot_y1 = (kpt1[46] +kpt1[49])/2
    foot_y2 = (kpt2[46] +kpt2[49])/2
    head_y1 = kpt1[1]
    head_y2 = kpt2[1]
    height1 = abs(foot_y1 - head_y1)
    height2 = abs(foot_y2 - head_y2)
    width1 = abs(kpt1[15] - kpt1[18])
    width2 = abs(kpt2[15] - kpt2[18])
    a1 = height1/width1
    a2 = height2/width2
    summation1 = 1.0/np.sum(confs1)
    summation2 = 0
    for i in range(0, len(coors1), 2):
        # x1, y1 = normalization(coors1[i], coors1[i+1], bbox1)
        # x2, y2 = normalization(coors2[i], coors2[i+1], bbox2)
        # summation2 += np.abs(x1 - x2 + y1 - y2)* confs1[i//2]
        summation2 += np.sqrt((coors1[i]-coors2[i])**2+ (coors1[i+1]-coors2[i+1])**2)* confs1[i//2]
    summation = summation1 * summation2 #* abs(a1-a2)
    return summation

def kpts_call2(detections, trackers):
  kpts = detections[:, 6:]
  kpts_prev = trackers[:, 6:]
  hs = detections[:, 3]- detections[:, 1]
  hs_prev = trackers[:, 3] - trackers[:, 1]
  kpt_matrix = np.zeros((len(kpts), len(kpts_prev)))
  # for i in range(num_people):
  #   kpt_matrix.append([])
  #   for j in range(num_people):
  #     kpt_matrix[i].append(0)
  minn = 999
  maxx = 0
  if len(kpts) == 0:
    return np.empty((0,2), dtype=int), np.arange(len(kpts)), np.empty((0, 5), dtype=int)
  if len(kpts) == 1 and len(kpts_prev) == 1:
    return np.array([[1]], dtype = np.float64)
  for i in range(len(kpts)):
    kpt = kpts[i]
    for j in range(len(kpts_prev)):
      kpt_prev = kpts_prev[j]
      #kpt_matrix[i][j] = 1/cal_pose_simi(kpt, kpt_prev)
      kpt_matrix[i][j] = np.exp(-cal_pose_simi2(kpt, kpt_prev)*2)
  #     if kpt_matrix[i][j] > maxx:
  #       maxx = kpt_matrix[i][j]
  #     if kpt_matrix[i][j] < minn:
  #       minn = kpt_matrix[i][j]
  # for i in range(len(kpts)):
  #   for j in range(len(kpts_prev)):
  #     kpt_matrix[i][j] = (kpt_matrix[i][j] - minn ) / (maxx - minn)
  return kpt_matrix

#------------------------------------------------------------------------------
def associate_detections_to_trackers(detections,trackers ,iou_threshold = 0.2):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,57),dtype=int)

  kpts = detections[:, 6:]
  kpts_prev = trackers[:, 6:]
  iou_matrix = iou_batch(detections, trackers)
  cos_matrix = kpts_call(kpts, kpts_prev)
  oks_matrix = kpts_call1(detections, trackers)
  kpt_matrix = kpts_call2(detections, trackers)
  for i in range(len(iou_matrix)):
    for j in range(len(iou_matrix[0])):
      if iou_matrix[i][j] == 0:
        cos_matrix[i][j] = 0
        oks_matrix[i][j] = 0
        kpt_matrix[i][j] = 0
  #print(f'iou = {iou_matrix}\n  kpt = {cos_matrix}')
  #print(type(iou_matrix[0][0]))
  combine_matrix = 0.8* iou_matrix + 0.2*kpt_matrix
  
  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-combine_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 57))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1    
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 57))
  
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[0:5] = [pos[0], pos[1], pos[2], pos[3], 0]
      #------
      trk[6:] = self.trackers[t].kpt
      #------
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], 0:5], dets[m[0], 6:])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,0:5], dets[i, 6:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,57))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")

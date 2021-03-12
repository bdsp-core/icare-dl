#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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


import pdb
import datetime
import time
from collections import Counter#, deque
import os
import os.path
import pickle
import sys
import subprocess
import scipy
#import matlab.engine
import h5py
import hdf5storage as hs
import scipy.io as sio
import numpy as np
import pandas as pd
#from read_delirium_data import *
from segment_EEG import *
from segment_EEG_without_detection import *
#from mne import *
import mne  as mne
import math

#from extract_features_parallel import *
#MATLAB_BIN_PATH = '/home/sunhaoqi/matlab/bin/matlab'
#EEGLAB_DIRECTORY = '/home/sunhaoqi/eeglab14_1_1b'
#MATLAB_CODE_PATH = 'rej_muscle_artifact.m'


Fs = 100.
#assess_time_before = 1800  # [s]
#assess_time_after = 1800  # [s]
window_time = 5  # [s]
window_step = 5  # [s]
#sub_window_time = 5  # [s] for calculating features
#sub_window_step = 1  # [s]
start_end_remove_window_num = 0
amplitude_thres = 500#500  # [uV]
line_freq = 60.  # [Hz]
bandpass_freq = [0.5, 30.]  # [Hz]
tostudy_freq = [0.5, 30.]  # [Hz]
#available_channels = ['C3', 'C4', 'O1', 'O2', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FZ', 'FP1', 'FP2', 'FPZ', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']#BWH
available_channels = ['EEG Fp1-Ref1', 'EEG F3-Ref1', 'EEG C3-Ref1', 'EEG P3-Ref1', 'EEG F7-Ref1', 'EEG T3-Ref1', 'EEG T5-Ref1', 'EEG O1-Ref1', 'EEG Fz-Ref1', 'EEG Cz-Ref1', 'EEG Pz-Ref1', 'EEG Fp2-Ref1',  'EEG F4-Ref1', 'EEG C4-Ref1', 'EEG P4-Ref1', 'EEG F8-Ref1', 'EEG T4-Ref1', 'EEG T6-Ref1', 'EEG O2-Ref1']  # UTW 
#available_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2',  'F4', 'C4', 'P4', 'F8', 'T8', 'P8', 'O2']
#available_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2',  'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2'] # MGH BWH ULB
bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',  'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
#available_channels = ['EEGFP1_', 'EEGFP2_', 'EEGFPZ_', 'EEGF7__', 'EEGF8__']
#eeg_channels = ['C3', 'C4', 'O1', 'O2', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FZ', 'FP1', 'FP2', 'FPZ', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']#['Fp1-F7','Fp2-F8','Fp1-Fp2','F7-F8']#'Fp1','Fp2','F7','F8',
#algorithms = ['cnn_lstm_ae', 'lstm', 'dnn_clf', 'dnn_ord', 'moe_dnn']#'RandomForest','SVM','ELM']'blr', 'dnn_reg', 'logreg', 
random_state = 1
#normal_only = False
#labeled_only = False

seg_mask_explanation = np.array([
    'normal',
    'NaN in EEG', #_[1,3] (append channel ids)
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'overly high/low total power',
    'muscle artifact',
    'multiple assessment scores',
    'spurious spectrum',
    'fast rising decreasing',
    '1Hz artifact',])

if __name__=='__main__':
    #"""
    ##################
    # use data_list_paths to specify what data to use
    # data_list.txt:
    # data_path    spec_path    feature_path    state
    # eeg1.mat     specs1.mat   Features1.mat   good
    # ...
    # note: eeg segments are contained in features
    ##################
    #file = "D:\\Research\\Cardiac_arrest_EEG\\EEG_weiLong_arrest_test_artifact\\bwh_7_1_2_20130201_205011.edf"
#    file_path = "D:\\Research\\Cardiac_arrest_EEG\\EEG_weiLong_arrest_test_artifact\\forWeilong\\"
#    save_path = "D:\\Research\\Cardiac_arrest_EEG\\EEG_weiLong_arrest_test_artifact\\Preprocessed\\"
    file_path = "Z:\\Projects\\CARDIAC_ARREST_DATA\\Reconvert_natus\\iCARE_final_092018\\UTW\\"
    save_path = "Z:\\Projects\\Weilong\\Cardiac_arrest_EEG\\Preprocessed\\UTW\\"
    file_list = [f for f in os.listdir(file_path) if f.endswith('.edf')]

    #file_list = sorted(file_list)
    #import pdb;pdb.set_trace()
    #file_list = os.listdir(file_path)
    file_list = file_list[155:156]
    
    for ifile in file_list:
        file = file_path + ifile
        print(file)
        #import pdb;pdb.set_trace()
#        if os.path.isfile(save_path+ifile+'.mat'):
#            continue
#        else:
#            try:
            #data = mne.io.read_raw_edf(file,preload=True)
        data = mne.io.read_raw_edf(file,stim_channel=None,exclude='EDF Annotations',preload=True)
        raw_data = data.get_data(picks=range(23))
        info = data.info
        fs = info['sfreq']
        #raw_data = scipy.signal.resample(raw_data, int(math.floor(raw_data.shape[1]*Fs/fs)),axis=1)
        raw_data = scipy.signal.resample_poly(raw_data, Fs, fs, axis=1)
        #raw_data = mne.filter.resample(raw_data, down=fs/Fs, npad='auto')
        
        #import pdb;pdb.set_trace()
        raw_data = raw_data*10e5 # V->uV
        
        channels = data.ch_names
        channels = [x.upper() for x in channels]
        chan_index = list()
        for chNo in available_channels:
            chan_index.append(channels.index(chNo.upper()))
        raw_data = raw_data[chan_index,:]
        
        
        
        ## Bipolar reference
        bipolar_data = np.zeros((18,raw_data.shape[1]))
        bipolar_data[8,:] = raw_data[0,:] - raw_data[1,:]; # Fp1-F3
        bipolar_data[9,:] = raw_data[1,:] - raw_data[2,:]; # F3-C3
        bipolar_data[10,:] = raw_data[2,:] - raw_data[3,:]; # C3-P3
        bipolar_data[11,:] = raw_data[3,:] - raw_data[7,:]; # P3-O1
    
        bipolar_data[12,:] = raw_data[11,:] - raw_data[12,:]; # Fp2-F4
        bipolar_data[13,:] = raw_data[12,:] - raw_data[13,:]; # F4-C4
        bipolar_data[14,:] = raw_data[13,:] - raw_data[14,:]; # C4-P4
        bipolar_data[15,:] = raw_data[14,:] - raw_data[18,:]; # P4-O2
    
        bipolar_data[0,:] = raw_data[0,:] - raw_data[4,:];  # Fp1-F7
        bipolar_data[1,:] = raw_data[4,:] - raw_data[5,:]; # F7-T3
        bipolar_data[2,:] = raw_data[5,:] - raw_data[6,:]; # T3-T5
        bipolar_data[3,:] = raw_data[6,:] - raw_data[7,:]; # T5-O1
    
        bipolar_data[4,:] = raw_data[11,:] - raw_data[15,:]; # Fp2-F8
        bipolar_data[5,:] = raw_data[15,:] - raw_data[16,:]; # F8-T4
        bipolar_data[6,:] = raw_data[16,:] - raw_data[17,:]; # T4-T6
        bipolar_data[7,:] = raw_data[17,:] - raw_data[18,:]; # T6-O2
    
        bipolar_data[16,:] = raw_data[8,:] - raw_data[9,:];   # Fz-Cz
        bipolar_data[17,:] = raw_data[9,:] - raw_data[10,:]; # Cz-Pz
        
        ## save 5s monopolar/bipolar epoches using notch/band pass/artifact detection/resampling
        segs_monpolar = segment_EEG_without_detection(raw_data,available_channels,window_time, window_step, Fs,
                            notch_freq=line_freq, bandpass_freq=bandpass_freq,
                            to_remove_mean=False, amplitude_thres=amplitude_thres, n_jobs=-1, start_end_remove_window_num=start_end_remove_window_num)
        del raw_data
        segs_, bs_, seg_start_ids_, seg_mask, specs_, freqs_ = segment_EEG(bipolar_data,bipolar_channels,window_time, window_step, Fs,
                            notch_freq=line_freq, bandpass_freq=bandpass_freq,
                            to_remove_mean=False, amplitude_thres=amplitude_thres, n_jobs=-1, start_end_remove_window_num=start_end_remove_window_num)
        
        if len(segs_) <= 0:
            raise ValueError('No segments')
            
        seg_mask2 = map(lambda x:x.split('_')[0], seg_mask)
        sm = Counter(seg_mask2)
        for ex in seg_mask_explanation:
            if ex in sm:
                print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))
                
        if segs_.shape[0]<=0:
            raise ValueError('No EEG signal')
            if segs_.shape[1]!=len(bipolar_channels):
                raise ValueError('Incorrect #chanels')
        
        fd = os.path.split(save_path)[0]
        if not os.path.exists(fd):
            os.mkdir(fd)
        res = {'EEG_segs_bipolar':segs_.astype('float16'),
               'EEG_segs_monopolar':segs_monpolar.astype('float16'),
               'EEG_specs':specs_.astype('float16'),
               'burst_suppression':bs_.astype('float16'),
               'EEG_frequency':freqs_,
               'seg_start_ids':seg_start_ids_,
               'Fs':Fs,
               'seg_masks':seg_mask,
               'channel_names':bipolar_channels}
        sio.savemat(save_path+ifile, res, do_compression=True)
                
#            except Exception as e:
#                continue
            
#    
#    data_list_paths = ['data/data_list.txt']
#    subject_files = np.zeros((0,5))
#    for data_list_path in data_list_paths:
#        subject_files = np.r_[subject_files, np.loadtxt(data_list_path, dtype=str, delimiter='\t', skiprows=1)]
#    subject_files = subject_files[subject_files[:,4]=='good',:4]
#    patient_ids = np.array([[x for x in xx.split('/') if x.startswith('icused')][0] for xx in subject_files[:,0]])
#    t0s = np.array([datenum(t0str, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True) for t0str in subject_files[:,2]])
#    t1s = np.array([datenum(t1str, '%Y-%m-%d %H:%M:%S.%f', return_seconds=True) for t1str in subject_files[:,3]])
#    """
#    # get the recording interval distribution
#    dists = []
#    for pid in np.unique(patient_ids):
#        tt0 = t0s[patient_ids==pid]
#        tt1 = t1s[patient_ids==pid]
#        ids = np.argsort(tt0)
#        tt0 = tt0[ids]
#        tt1 = tt1[ids]
#        assert np.all(np.diff(tt1)>0)
#        assert np.all(tt1-tt0>0)
#        dists.extend(tt0[1:]-tt1[:-1])
#    plt.hist(np.log1p(dists),bins=50);plt.show()
#    """
#    record_num = subject_files.shape[0]
#    
#    subject_err_path = 'data/err_subject_reason.txt'
#    if os.path.isfile(subject_err_path):
#        err_subject_reason = []
#        with open(subject_err_path,'r') as f:
#            for row in f:
#                if row.strip()=='':
#                    continue
#                i = row.split(':::')
#                err_subject_reason.append([i[0].strip(), i[1].strip()])
#        err_subject = [i[0] for i in err_subject_reason]
#    else:
#        err_subject_reason = []
#        err_subject = []
#
#    all_rass_times = np.loadtxt('data/rass_times.txt', dtype=str, delimiter='\t', skiprows=1)
#    all_camicu_times = pd.read_csv('data/vICU_Sed_CamICU.csv', sep=',')
#    for si in range(record_num):
#        data_path = subject_files[si,0]
#        feature_path = subject_files[si,1]
#        t0 = t0s[si]
#        t1 = t1s[si]
#        #assert t1 == t0+res['data'].shape[1]*1./Fs
#        patient_id = patient_ids[si]
#        subject_file_name = os.path.join(patient_id, data_path.split('/')[-1])
#        if subject_file_name in err_subject:
#            continue
#        if os.path.isfile(feature_path):
#            print('\n[(%d)/%d] %s %s'%(si+1,record_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
#        else:
#            print('\n[%d/%d] %s %s'%(si+1,record_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
#            try:
#                # check and load dataset
#                res = read_delirium_mat(data_path, channel_names=available_channels)#, with_time=False)
#                if res['Fs']<Fs-1 or res['Fs']>Fs+1:
#                    raise ValueError('Fs is not %gHz.'%Fs)
#                #if res['data'].shape[1]<Fs*3600*0.5:
#                #    raise ValueError('Recording is less than 30min.')
#                #dt = (t1-t0)-res['data'].shape[1]*1./Fs
#                #if np.abs(dt) >= 300:
#                #    raise TypeError('Miss-matched t0 and t1 in %s: %gs'%(subject_file_name,dt))
#
#                # segment EEG
#                segs_, bs_, labels_, assessment_times_, seg_start_ids_, seg_mask, specs_, freqs_ = segment_EEG(res['data'],
#                        all_rass_times[all_rass_times[:,1]==patient_id,:],
#                        all_camicu_times[all_camicu_times.PatientID==patient_id].values,
#                        assess_time_before, assess_time_after,
#                        [t0,t1], t0s[patient_ids==patient_id], t1s[patient_ids==patient_id], window_time, window_step, Fs,
#                        notch_freq=line_freq, bandpass_freq=bandpass_freq,
#                        to_remove_mean=False, amplitude_thres=amplitude_thres, n_jobs=-1, start_end_remove_window_num=start_end_remove_window_num)
#                if len(segs_) <= 0:
#                    raise ValueError('No segments')
#                #bs_ = (bs_<=5).sum(axis=2)/1000.
#                
#                if labeled_only:
#                    raise NotImplementedError('labeled_only==True')
#                    good_ids = np.where(np.logical_not(np.isnan(labels_)))[0]
#                    segs_ = segs_[good_ids]
#                    bs_ = bs_[good_ids]
#                    labels_ = labels_[good_ids]
#                    assessment_times_ = [assessment_times_[ii] for ii in good_ids]
#                    seg_start_ids_ = seg_start_ids_[good_ids]
#                    seg_mask = [seg_mask[ii] for ii in good_ids]
#                    specs_ = specs_[good_ids]
#                    #specs_matlab = specs_matlab[:,:,good_ids]
#                
#                """
#                # muscle artifacts
#                specs_matlab = 10*np.log(specs_.T)
#                specs_matlab[np.isinf(specs_matlab)] = np.nan
#                specs_matlab_mean = np.nanmean(specs_matlab, axis=2, keepdims=True)
#                specs_matlab = specs_matlab - specs_matlab_mean
#                sio.savemat('segs.mat', {'segs':segs_.transpose(1,2,0), 'Fs':Fs, 'specs':specs_matlab})#, 'specs_orig':specs_.T
#                with open('matlab_output.txt','w') as ff:
#                    subprocess.check_call([MATLAB_BIN_PATH, '<', MATLAB_CODE_PATH], stdout=ff)
#                muscle_rej_ch2d = sio.loadmat('rej.mat')['rejE'].T==1  # (#sample, #channel)            
#                muscle_rej_ch1d = np.where(np.any(muscle_rej_ch2d,axis=1))[0]
#                for i in muscle_rej_ch1d:
#                    seg_mask[i] = '%s_%s'%(seg_mask_explanation[7], np.where(muscle_rej_ch2d[i])[0])
#                """
#                
#                #segs_ = segs_[:3]
#                #features_, feature_names = extract_features(segs_, Fs, sub_window_time, tostudy_freq,
#                #                                                sub_window_time, sub_window_step, data_path,
#                #                                                seg_start_ids_,
#                #                                                return_feature_names=True, n_jobs=-1)#, specs_, freqs_
#
#                print('\n%s\n'%Counter(labels_[np.logical_not(np.isnan(labels_))]))
#                #bsp = features_[:,-1]       
#                #print('BSP\nmax: %g\nmin: %g\n'%(np.max(bsp),np.min(bsp)))
#                #if np.max(bsp)-np.min(bsp)<=1e-3:
#                #    raise ValueError('Flat BSP')
#                                        
#                #features_[np.isinf(features_)] = np.nan
#                #nan1d = np.where(np.any(np.isnan(features_),axis=1))[0]
#                #for i in nan1d:
#                #    seg_mask[i] = seg_mask_explanation[4]
#                
#                #seg_mask = np.array(seg_mask)
#                seg_mask2 = map(lambda x:x.split('_')[0], seg_mask)
#                sm = Counter(seg_mask2)
#                for ex in seg_mask_explanation:
#                    if ex in sm:
#                        print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))
#                
#                if normal_only:
#                    good_ids = np.where(np.array(seg_mask)=='normal')[0]
#                    segs_ = segs_[good_ids]
#                    bs_ = bs_[good_ids]
#                    labels_ = labels_[good_ids]
#                    assessment_times_ = [assessment_times_[ii] for ii in good_ids]
#                    seg_start_ids_ = seg_start_ids_[good_ids]
#                    seg_mask = [seg_mask[ii] for ii in good_ids]
#                    specs_ = specs_[good_ids]
#                    #specs_matlab = specs_matlab[:,:,good_ids]
#                
#                if segs_.shape[0]<=0:
#                    raise ValueError('No EEG signal')
#                if segs_.shape[1]!=len(eeg_channels):
#                    raise ValueError('Incorrect #chanels')
#
#            except Exception as e:
#                """
#                err_info = e.message.split('\n')[0].strip()
#                print('\n%s.\nSubject %s is IGNORED.\n'%(err_info,subject_file_name))
#                err_subject_reason.append([subject_file_name,err_info])
#                err_subject.append(subject_file_name)
#
#                with open(subject_err_path,'a') as f:
#                    msg_ = '%s::: %s\n'%(subject_file_name,err_info)
#                    f.write(msg_)
#                """
#                continue
#
#            fd = os.path.split(feature_path)[0]
#            if not os.path.exists(fd):
#                os.mkdir(fd)
#            res = {'EEG_segs':segs_.astype('float32'),
#                'EEG_specs':specs_.astype('float32'),
#                'burst_suppression':bs_.astype('float32'),
#                'EEG_frequency':freqs_,
#                #'EEG_features':features_,
#                #'feature_names':feature_names
#                't0':subject_files[si,2],
#                't1':subject_files[si,3],
#                'labels':labels_,
#                'assess_times':assessment_times_,
#                'seg_start_ids':seg_start_ids_,
#                'subject':subject_file_name,
#                'Fs':Fs}
#            if not normal_only:
#                res['seg_masks'] = seg_mask
#            sio.savemat(feature_path, res, do_compression=True)
#            res = sio.loadmat(feature_path)
#            os.remove(feature_path)
#            time.sleep(1)
#            hs.savemat(feature_path, res)
#                
#

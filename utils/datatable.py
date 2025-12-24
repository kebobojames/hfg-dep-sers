import os
import pickle
import numpy as np
import re

class DataTableUrine1115(object):
    def __init__(self, data_path, cache_fold_path="./241115_urine_cache", prefix = "novoltage"):
        assert prefix in ["center", "surroundings", "novoltage"]
        
        cache_file_path = os.path.join(cache_fold_path, f"{prefix}_urine_data.pkl")
        if os.path.isfile(cache_file_path):
            self.data_np = pickle.load(open(cache_file_path, "rb"))
            self.label = pickle.load(open(os.path.join(cache_fold_path, f"{prefix}_urine_label.pkl"), "rb"))
            self.raman_shift = pickle.load(open(os.path.join(cache_fold_path, f"{prefix}_urine_shift.pkl"), "rb"))
            self.subject = pickle.load(open(os.path.join(cache_fold_path, f"{prefix}_urine_subject.pkl"), "rb"))
        else:
            out_data_list = []
            out_label_list = []
            out_subject_no_list = []

            files = [(dirpath, filenames) for dirpath, dirnames, filenames in os.walk(data_path) if dirnames == []]
            files.sort(key = lambda x: x[0])
            
            # label_class = 0
            for dirpath, filenames in files:    
                for file in filenames:
                    if not (file.startswith('Norm') or file.startswith('Lung') or file.startswith('Pan')): 
                        print(f'wrong name: {file} in {dirpath}')
                        continue

                    match = None
                    if file.startswith('Norm'):
                        match = re.search(r"P\d{2}", file)
                    elif file.startswith('Lung'):
                        match = re.search(r"S\d{3}|T\d{3}|T\d{2}", file)
                    else:
                        match = re.search(r"Pan\d{3}|Pan\d{2}", file)
                    
                    if not match:
                        print(f"can't find match: None")
                        continue

                    match = match.group()
                    
                    if (file.startswith('Norm') and not match in ['P29', 'P30', 'P31', 'P37', 'P38', 'P39', 'P45', 'P50', 'P25', 'P27', 'P52']) or \
                       (file.startswith('Lung') and not match in ['S002', 'S012', 'S017', 'S022', 'S023', 'S028', 'S031', 'T001', 'T01', 'S034', 'S035', 'T002']) or \
                       (file.startswith('Pan') and not match in ['Pan01', 'Pan14', 'Pan16', 'Pan17', 'Pan22', 'Pan25', 'Pan42', 'Pan56', 'Pan29', 'Pan029', 'Pan35', 'Pan49']):
                            print(f"can't find match: {file}, {match}")
                            continue
                    
                    out_subject_no_list.append(np.array([match]))

                    data_list_pre = []
                    raman_shift = []
                    with open(os.path.join(dirpath, file), "r") as file_obj:
                        single_flag = True
                        lines = file_obj.readlines()

                        if "FILETYPE" in lines[0]:
                            # assert "XYDATA=" in lines[19]
                            findstart = 0
                            try:
                                while True:
                                    if "XYDATA=" in lines[findstart]:
                                        break
                                    findstart += 1
                            except:
                                assert False, 'no XYDATA'

                            for i in range(findstart + 1): lines.pop(0)
                            for line in lines:
                                tokens = line.strip().split(",")
                                raman_shift.append(tokens[0])
                                data_list_pre.append(tokens[1])
                                
                        else:
                            print("shouldn't be here")
                            raise ValueError
                            # single_flag = False
                            # for line in lines:
                            #     tokens = line.strip().split()
                            #     raman_shift.append(tokens[0])
                            #     data_list_pre.append(tokens[1:])

                        data_list_pre = np.array(data_list_pre, ndmin = 2, dtype = 'float64')
                        if (prefix != 'novoltage' and "Pancreatic 29" in dirpath and 'Pan29' in file):
                            print(dirpath, file, 'shifting!')
                            shifted = np.zeros_like(data_list_pre)
                            shifted[:, 1:] = data_list_pre[:, :-1]
                            data_list_pre = shifted
                        # if not single_flag: data_list_pre = np.transpose(data_list_pre)
                        
                    # label_out = [label_class] * np.shape(data_list_pre)[0]
                    label_out = [0 if file.startswith('Norm') else 1 if file.startswith('Lung') else 2] * np.shape(data_list_pre)[0]
                    label_out = np.array(label_out)

                    raman_shift_out = np.array(raman_shift, dtype = 'float64')

                    out_data_list.append(data_list_pre)
                    out_label_list.append(label_out)
                
                # label_class += 1

            out_data = np.concatenate(out_data_list, axis=0)
            out_label = np.concatenate(out_label_list, axis=0)
            out_subject = np.concatenate(out_subject_no_list, axis=0)

            print(raman_shift_out)
            print(out_data.shape)
            print(out_label.shape)
            print(np.unique(out_label))

            pickle.dump(out_data, open(cache_file_path, "wb"))
            pickle.dump(out_label, open(os.path.join(cache_fold_path, f"{prefix}_urine_label.pkl"), "wb"))
            pickle.dump(raman_shift_out, open(os.path.join(cache_fold_path, f"{prefix}_urine_shift.pkl"), "wb"))
            pickle.dump(out_subject, open(os.path.join(cache_fold_path, f"{prefix}_urine_subject.pkl"), "wb"))

            self.data_np = out_data
            self.label = out_label
            self.raman_shift = raman_shift_out
            self.subject = out_subject
            
    def return_np_raw(self, range=[550,1800]):
        idxs = np.where((self.raman_shift >= range[0]) & (self.raman_shift <= range[1]))[0]
        print(idxs)
        return (self.data_np[:, idxs], self.label, self.raman_shift[idxs], self.subject)

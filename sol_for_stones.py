import os
import psutil
import time

PROCNAME1 = "python"
PROCNAME2 = "python3"
PROCNAME3 = "python3.9"

####################

pids_before_1 = []
for proc in psutil.process_iter():
    if proc.name() == PROCNAME1 and proc.status() != "sleeping":
        pids_before_1 += [proc.pid]

print("pids before 1", pids_before_1)

##########

pids_before_2 = []
for proc in psutil.process_iter():
    if proc.name() == PROCNAME2:
        pids_before_2 += [proc.pid]

print("pids before 2", pids_before_2)

##########

pids_before_3 = []
for proc in psutil.process_iter():
    if proc.name() == PROCNAME3:
        pids_before_3 += [proc.pid]

print("pids before 3", pids_before_3)

####################
print('open file')
with open("/home/vskovoroda/all_models.txt", "r") as file:
    all_exec_lines = file.read().split("\n")

exec_const = "source miniconda3/bin/activate; conda activate tf-gpu; cd Stones; "

ind = 0
print('Start')
while ind < len(all_exec_lines):
    if os.path.isfile('/home/vskovoroda/Stones/'+all_exec_lines[ind].split()[3]):
        ind+=1
        print('continue with ind =',ind)
        continue
    time.sleep(5)
    
    pids_after_1 = []
    pids_after_2 = []
    pids_after_3 = []
    
    for proc in psutil.process_iter():
        if proc.name() == PROCNAME1 and proc.status() != "sleeping":
            pids_after_1 += [proc.pid]
        if proc.name() == PROCNAME2:
            pids_after_2 += [proc.pid]
        if proc.name() == PROCNAME3:
            pids_after_3 += [proc.pid]
            
    if len(pids_after_1) == 0:
        print('Запустил пятую')
        params = all_exec_lines[ind].split()
        exec_line = exec_const + f"nohup {PROCNAME1} train_model.py --save_fname={params[0]} --encoder={params[1]} --gpu=4 --bs={params[2]} > {params[3]} &"
        os.system(exec_line)
        ind+=1

    # ##########
    
    if len(pids_after_2) == 0:
        print('Запустил Четвёртую')
        params = all_exec_lines[ind].split()
        exec_line = exec_const + f"nohup {PROCNAME2} train_model.py --save_fname={params[0]} --encoder={params[1]} --gpu=3 --bs={params[2]} > {params[3]} &"
        os.system(exec_line)
        ind+=1

    #########
    
    if len(pids_after_3) == 0:
        print('Запустил Вторую')
        params = all_exec_lines[ind].split()
        exec_line = exec_const + f"nohup {PROCNAME3} train_model.py --save_fname={params[0]} --encoder={params[1]} --gpu=1 --bs={params[2]} > {params[3]} &"
        os.system(exec_line)
        ind+=1

print("fin")
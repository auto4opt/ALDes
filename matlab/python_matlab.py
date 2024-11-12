import matlab.engine
import matlab
import os

eng = matlab.engine.start_matlab()
# 此地址为test.m文件存放的地址
work_path = os.getcwd() + "\\matlab"
eng.cd(work_path)

a = matlab.double([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

#c = eng.get_per(a)
result = eng.sa_for_python()
print(result)
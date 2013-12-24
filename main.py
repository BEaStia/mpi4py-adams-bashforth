
from mpi4py import MPI

def factors():
    return [[-1,2,-5,3],[3,-4,6,-12],[9,-2,4,-4],[1,4,-3,1]]

def func_new(args,rank):
    result = 0
    for i in range(0, len(factors())):
        result+=factors()[rank][i]*args[i]
    return result

def func(args,y):
    from math import sqrt
    if (args.__class__.__name__=="list"):
        return args[0]*sqrt(y)
    else:
        return args*sqrt(y)

def eiler_step(h, args):
    old_args = args
    var_set = []
    for i in range(0,len(factors())):
        val = old_args[i] + h*func_new(args,i)
        var_set.append(val)
    return var_set

def eiler(startT,endT,h,args):
    all_args = [args]
    nmax=int((endT-startT)/h +1)
    for n in range(1,nmax):
        temp_args = all_args[n-1]
        temp_args = eiler_step(h,temp_args)
        all_args.append(temp_args)
    return all_args

def Adams2(startT,endT,h,args):
    result = eiler(startT,startT+h,h,args)
    nmax=int((endT-startT)/h +1)
    for n in range(len(result),nmax):
        prev_args_2 = result[len(result)-2]
        prev_args_1 = result[len(result)-1]
        var_set = []
        for i in range(0,len(factors())):
            val = prev_args_1[i] + h*(3*func_new(prev_args_1,i)-1*func_new(prev_args_2,i))/2
            var_set.append(val)
        result.append(var_set)
    return result

def Adams3(startT,endT,h,args):
    result = Adams2(startT,startT+h*2,h,args)
    nmax=int((endT-startT)/h +1)
    for n in range(len(result),nmax):
        prev_args = result[-3:]
        var_set = []
        for i in range(0,len(factors())):
            val = prev_args[-1][i] + (23*func_new(prev_args[-1],i)-16*func_new(prev_args[-2],i)+5*func_new(prev_args[-3],i))/12*h
            var_set.append(val)
        result.append(var_set)
    return result

def Adams3MPI(startT,endT,h,args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nmax = 0
    prev_args = []
    if(rank==0):
        result = Adams2(startT,startT+h*2,h,args)
        nmax=int((endT-startT)/h )
    nmax = comm.bcast(nmax,root=0)
    for n in range(3,nmax):
        if rank==0:
            vars_for_main = []
            prev_args = result[-3:]
            var_set = [None]*len(factors())
            if(size<len(prev_args)):
                vars_for_main = range(size,len(prev_args))
        prev_args = comm.bcast(prev_args,root=0)
        print("prev_args:",prev_args)
        val = prev_args[-1][rank]+(23*func_new(prev_args[-1],rank)-16*func_new(prev_args[-2],rank)+5*func_new(prev_args[-3],rank))/12*h
        print "at rank ",rank,' number',val
        if rank!=0:
            tag = 10+rank
            print("sending from ",rank," with tag:",tag,val)
            comm.send(val,dest=0,tag=tag)
        if rank==0:
            var_set[rank] = val
            if(len(vars_for_main)>0):
                for i in vars_for_main :
                    val = prev_args[-1][i] + (23*func_new(prev_args[-1],i)-16*func_new(prev_args[-2],i)+5*func_new(prev_args[-3],i))/12*h
                    var_set[i] = val
            else:
                print("Without any unefficient threads")
            for i in range(1,size):
                print("waiting from ",i," with tag:",i+10)
                tag = i+10
                obj = comm.recv(source=i, tag=tag)
                var_set[i] = obj
            result.append(var_set)
        comm.Barrier()

    if(rank==0):
        return result
    else:
        return []

def Adams5(startT,endT,h,args):
    result = Adams3(startT,startT+h*4,h,args)
    nmax=int((endT-startT)/h +1)
    for n in range(len(result),nmax):
        prev_args_5 = result[len(result)-5]
        prev_args_4 = result[len(result)-4]
        prev_args_3 = result[len(result)-3]
        prev_args_2 = result[len(result)-2]
        prev_args_1 = result[len(result)-1]
        var_set = []
        for i in range(0,len(factors())):
            val = prev_args_1[i] + (1901*func_new(prev_args_1,i)-1387*2*func_new(prev_args_2,i)+109*24*func_new(prev_args_3,i)-637*2*func_new(prev_args_4,i)+251*func_new(prev_args_5,i))/720*h
            var_set.append(val)
        result.append(var_set)
    return result

#
# def many_predict_2(h,args,start):
#     result = []
#     for i in range(0,len(start)):
#         val = start[i] + h*func_new(args,i)
#         result.append(val)
#     return result
#
# def many_correct_2(h,args,start,result,new_args):
#     corrected_result = []
#     for i in range(0,len(start)):
#         val = start[i] + h*(func_new(args,i)+func_new(new_args,i))/2
#         corrected_result.append(val)
#     return corrected_result
#
# def many_Predict_corrector_2(h,startT,endT,args,start):
#     yi=[start]
#     xi=[args]
#     nmax=int((endT-startT)/h +1)
#     for n in range(1,nmax,1):
#         cur_args = xi[n-1]
#         vals = yi[n-1]
#         predictor = many_predict_2(h,args,start)
#         new_args = cur_args
#         for i in range(0,len(cur_args)):
#             new_args[i] += h
#         y_n = correct_2(h,args,vals,new_args)
#         yi.append(y_n)
#         xi.append([args[0]+h])
#     return [ti,yi]
#
#
#
# def predict_2(h,y,args):
#     y_new = y + h*func(args,y)
#     return y_new
#
# def correct_2(h,y,y_new,args,new_args):
#     y_n = y + h*(func(args,y)+func(new_args,y_new))/2
#     return y_n
#
# def realAdams2(h,startT,endT,args,y):
#     result = Predict_corrector_2(h,startT,startT+h,[args],y)
#     yi = result[1]
#     ti = result[0]
#     nmax=int((endT-startT)/h +1)
#     for n in range(len(yi),nmax,1):
#         y_n_2 = yi[n-2]
#         y_n_1 = yi[n-1]
#         t_2 = ti[n-2][0]
#         t_1 = ti[n-1][0]
#         y_n = y_n_1 + h*(3*func(t_1,y_n_1)-func(t_2,y_n_2))/2
#         yi.append(y_n)
#         ti.append([t_1+h])
#     return [ti,yi]
#
# def realAdams3(h,startT,endT,args,y):
#     result = Predict_corrector_2(h,startT,startT+h*2,[args],y)
#     yi = result[1]
#     ti = result[0]
#     print yi
#     print ti
#     nmax=int((endT-startT)/h +1)
#     for n in range(3,nmax,1):
#         y_n_3 = yi[n-3]
#         y_n_2 = yi[n-2]
#         y_n_1 = yi[n-1]
#         t_3 = ti[n-3][0]
#         t_2 = ti[n-2][0]
#         t_1 = ti[n-1][0]
#         fun_1 = 23*func(t_1,y_n_1)*h/12
#         fun_2 = -16*func(t_2,y_n_2)*h/12
#         fun_3 = 5*func(t_3,y_n_3)*h/12
#         y_n = y_n_1 + fun_1 + fun_2 + fun_3
#         yi.append(y_n)
#         ti.append([t_1+h])
#     return [ti,yi]
#
# def many_realAdams3(h,startT,endT,args,y):
#     result = Predict_corrector_2(h,startT,startT+h*2,[args],y)
#     yi = result[1]
#     ti = result[0]
#     print yi
#     print ti
#     nmax=int((endT-startT)/h +1)
#     for n in range(3,nmax,1):
#         y_n_3 = yi[n-3]
#         y_n_2 = yi[n-2]
#         y_n_1 = yi[n-1]
#         t_3 = ti[n-3][0]
#         t_2 = ti[n-2][0]
#         t_1 = ti[n-1][0]
#         fun_1 = 23*func(t_1,y_n_1)*h/12
#         fun_2 = -16*func(t_2,y_n_2)*h/12
#         fun_3 = 5*func(t_3,y_n_3)*h/12
#         y_n = y_n_1 + fun_1 + fun_2 + fun_3
#         yi.append(y_n)
#         ti.append([t_1+h])
#     return [ti,yi]
#
# def mpy_realAdams3(h,startT,endT,args,y):
#     result = Predict_corrector_2(h,startT,startT+h*2,[args],y)
#     yi = result[1]
#     ti = result[0]
#     print yi
#     print ti
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     nmax=int((endT-startT)/h +1)
#     for n in range(3,nmax,1):
#         y_n = None
#         if(rank==0):
#             y_n_3 = yi[n-3]
#             y_n_2 = yi[n-2]
#             y_n_1 = yi[n-1]
#             t_3 = ti[n-3][0]
#             t_2 = ti[n-2][0]
#             t_1 = ti[n-1][0]
#             comm.send({'y':y_n_1,'t':t_1}, dest=1, tag=11)
#             comm.send({'y':y_n_2,'t':t_2}, dest=2, tag=12)
#             comm.send({'y':y_n_3,'t':t_3}, dest=3, tag=13)
#             fun_1=comm.recv(source=1,tag=11)
#             fun_2=comm.recv(source=2,tag=21)
#             fun_3=comm.recv(source=3,tag=31)
#             y_n = y_n_1 + fun_1 + fun_2 + fun_3
#
#         if(rank==1):
#             data=comm.recv(source=0,tag=11)
#             fun_1 = 23*func(data['t'],data['y'])*h/12
#             comm.send(fun_1,dest=0,tag=11)
#         if(rank==2):
#             data=comm.recv(source=0,tag=12)
#             fun_2 = -16*func(data['t'],data['y'])*h/12
#             comm.send(fun_2,dest=0,tag=21)
#         if(rank==3):
#             data=comm.recv(source=0,tag=13)
#             fun_3 = 5*func(data['t'],data['y'])*h/12
#             comm.send(fun_3,dest=0,tag=31)
#         y_n = comm.bcast(y_n,root=0)
#         yi.append(y_n)
#         ti.append([ti[n-1][0]+h])
#     return [ti,yi]
#
# def realAdams4(h,startT,endT,args,y):
#     result = Predict_corrector_2(h,startT,startT+h*3,[args],y)
#     yi = result[1]
#     ti = result[0]
#     print yi
#     print ti
#     nmax=int((endT-startT)/h +1)
#     for n in range(4,nmax,1):
#         y_n_4 = yi[n-4]
#         y_n_3 = yi[n-3]
#         y_n_2 = yi[n-2]
#         y_n_1 = yi[n-1]
#         t_4 = ti[n-4][0]
#         t_3 = ti[n-3][0]
#         t_2 = ti[n-2][0]
#         t_1 = ti[n-1][0]
#         fun_1 = func(t_1,y_n_1)*h*55/24
#         fun_2 = func(t_2,y_n_2)*h*(-59)/24
#         fun_3 = func(t_3,y_n_3)*h*37/24
#         fun_4 = func(t_4,y_n_4)*h*(-9)/24
#         y_n = y_n_1 + fun_1 + fun_2 + fun_3 + fun_4
#         yi.append(y_n)
#         ti.append([t_1+h])
#     return [ti,yi]
#
# def realAdams5(h,startT,endT,args,y):
#     result = Predict_corrector_2(h,startT,startT+h*4,[args],y)
#     yi = result[1]
#     ti = result[0]
#     print yi
#     print ti
#     nmax=int((endT-startT)/h +1)
#     for n in range(5,nmax,1):
#         y_n_5 = yi[n-5]
#         y_n_4 = yi[n-4]
#         y_n_3 = yi[n-3]
#         y_n_2 = yi[n-2]
#         y_n_1 = yi[n-1]
#         t_5 = ti[n-5][0]
#         t_4 = ti[n-4][0]
#         t_3 = ti[n-3][0]
#         t_2 = ti[n-2][0]
#         t_1 = ti[n-1][0]
#         fun_1 = func(t_1,y_n_1)*h*1901/720
#         fun_2 = func(t_2,y_n_2)*h*(-1387)/360
#         fun_3 = func(t_3,y_n_3)*h*109/30
#         fun_4 = func(t_4,y_n_4)*h*(-637)/360
#         fun_5 = func(t_5,y_n_5)*h*251/720
#         y_n = y_n_1 + fun_1 + fun_2 + fun_3 + fun_4 + fun_5
#         yi.append(y_n)
#         ti.append([t_1+h])
#     return [ti,yi]
#
# def Predict_corrector_2(h,startT,endT,args,y):
#     yi=[y]
#     ti=[args]
#     nmax=int((endT-startT)/h +1)
#     for n in range(1,nmax,1):
#         args = ti[n-1]
#         y = yi[n-1]
#         predictor = predict_2(h,y,args)
#         new_args = [args[0]+h]
#         y_n = correct_2(h,y,predictor,args,new_args)
#         yi.append(y_n)
#         ti.append([args[0]+h])
#     return [ti,yi]
#
#
# def RK4(yp,tmin,tmax,y0,t0,dt):
#     yi=[y0]
#     ti=[t0]
#     nmax=int( (tmax-tmin)/dt +1)
#     print nmax,"iterations"
#     for n in range(1,nmax,1):
#         tn=ti[n-1]
#         yn=yi[n-1]
#         dy1=dt*yp( tn,        yn)
#         dy2=dt*yp( tn+dt/2.0, yn+dy1/2.0)
#         dy3=dt*yp( tn+dt/2.0, yn+dy2/2.0)
#         dy4=dt*yp( tn+dt,     yn+dy3)
#         yi.append( yn+(1.0/6.0)*(dy1+2.0*dy2+2.0*dy3+dy4) )
#         ti.append(tn+dt)
#     return [ti,yi]

from datetime import datetime
time_before_adams = datetime.now()
h = 0.001
startT = 0.0
endT = 10.0
#y0 = 1.0
#var = eiler(startT,endT,h,[1,1,1,1])
#var = Adams3(startT,endT,h,[1,1,1,1])
var = Adams3MPI(startT,endT,h,[1,1,1,1])
time_between_methods = datetime.now()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if(rank==0):
    for i in range(0,len(var)):
        print(i,var[i])
#var = Adams3(startT,endT,h,[1,1,1,1])
#time_after_all = datetime.now()
#print (time_between_methods-time_before_adams).microseconds,"ms adams"
#print (time_after_all-time_between_methods).microseconds,"ms adams"

#adams = mpy_realAdams3(h,startT,endT,startT,y0)
#adams = realAdams3(h,startT,endT,startT,y0)
#time_between_methods = datetime.now()
#rk4 = RK4(func,startT,endT,y0,startT,h)
#time_after_rk4 = datetime.now()
#print (time_between_methods - time_before_adams).microseconds,"ms adams"
#print (time_after_rk4 - time_between_methods).microseconds,"ms rk4"
#t = adams[0]
#y = rk4[1]
#y_2 = adams[1]
#omm = MPI.COMM_WORLD
#ank = comm.Get_rank()
#if()
#for i in range(0,len(t),10):
#    print ("y(%3.3f)\t= %4.8f \t error:%4.8g \tvs\t y(%3.3f)\t= %4.8f \t error:%4.8g")%(t[i][0], y[i], abs( y[i]- ((t[i][0]**2 + 4.0)**2)/16.0 ), t[i][0], y_2[i], abs( y_2[i]- ((t[i][0]**2 + 4.0)**2)/16.0 ))



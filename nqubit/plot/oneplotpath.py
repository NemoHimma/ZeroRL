import numpy as np 
import matplotlib.pyplot as plt
import math
plt.rcParams['axes.unicode_minus']=False
T=1  #4500
Pi=3.14
t = np.linspace(0, T, 10000)
#= 1/T*t-(0.03)*np.sin(2*Pi*t/T)+0.03*np.sin(4*Pi*t/T)+0.02*np.sin(6*Pi*t/T)
#y= 1/T*t+(0.01)*np.sin(2*Pi*t/T)+0.01*np.sin(4*Pi*t/T)+0.04*np.sin(6*Pi*t/T)
#y = 1/T*t+-0.01374316*np.sin(1*Pi*t/T)+0.03760616*np.sin(2*Pi*t/T)+0.02280342*np.sin(3*Pi*t/T)+-0.02083123*np.sin(4*Pi*t/T)+0.00763241*np.sin(5*Pi*t/T)+0.00389999*np.sin(6*Pi*t/T)
#y = 1/T*t+0.0282776*np.sin(1*Pi*t/T)+0.09969209*np.sin(2*Pi*t/T)+-0.0010771*np.sin(3*Pi*t/T)+0.00284987*np.sin(4*Pi*t/T)+-0.02266757*np.sin(5*Pi*t/T)+0.00848344*np.sin(6*Pi*t/T)

# y = 1/T*t+0.04858864*np.sin(1*Pi*t/T)+0.09337428*np.sin(2*Pi*t/T)+-0.0*np.sin(3*Pi*t/T)\
# +0.05357208*np.sin(4*Pi*t/T)+0.01919774*np.sin(5*Pi*t/T)+0.02885251*np.sin(6*Pi*t/T)
# y = 1/T*t+-0.00654086*np.sin(1*Pi*t/T)+-0.0054546*np.sin(2*Pi*t/T)+0.02867583*np.sin(3*Pi*t/T)+0.03320241*np.sin(4*Pi*t/T)+-0.01144606*np.sin(5*Pi*t/T)+-0.04066887*np.sin(6*Pi*t/T)


#y = 1/T*t+0.00648197*np.sin(1*Pi*t/T)+-0.00400127*np.sin(2*Pi*t/T)+-0.01657386*np.sin(3*Pi*t/T)+-0.04133466*np.sin(4*Pi*t/T)+-0.02144199*np.sin(5*Pi*t/T)+0.00808055*np.sin(6*Pi*t/T)
#8 easy

a = np.array([-0.12872561  ,0.05571891,-0.01423914 , 0.02609146 ,-0.0182819 ,  0.03412344]) # 5
#a = np.array([0.25157502,  0.06189311 , 0.02930492 , 0.04284022 , 0.04045251, -0.00355106])
b = np.array([-0.04081619 , 0.0409642  , 0.01492415, -0.03345438 , 0.01596453 , 0.00974784]) # 6
c = np.array([-0.12606458 , 0.14979449, -0.0121935 , -0.12477024,  0.05985154 , 0.01539865]) # 7
d = np.array([-0.24434611 , 0.19208736 ,-0.10735239 , 0.07708072 ,-0.04869116 , 0.02009886]) # 9
e = np.array([-0.17647199 , 0.20449194, -0.13439653 , 0.05501544, -0.04193713,  0.02698592])  #11
y_1 =(1/T*t)**2+a[0]*np.sin(1*Pi*t/T)+a[1]*np.sin(2*Pi*t/T)+a[2]*np.sin(3*Pi*t/T)\
+a[3]*np.sin(4*Pi*t/T)+a[4]*np.sin(5*Pi*t/T)+a[5]*np.sin(6*Pi*t/T)

# y_2 = 1-(1/T*t)**2+a[3]*np.sin(1*Pi*t/T)+a[4]*np.sin(2*Pi*t/T)+a[5]*np.sin(3*Pi*t/T)\

# y = (1/T*t)**2+a[0]*np.sin(1*Pi*t/T)+a[1]*np.sin(2*Pi*t/T)+a[2]*np.sin(3*Pi*t/T)\
# +a[3]*np.sin(4*Pi*t/T)+a[4]*np.sin(5*Pi*t/T)+a[5]*np.sin(6*Pi*t/T)

# y_origin = (1/T*t)**2

y_2= (1/T*t)**2+b[0]*np.sin(1*Pi*t/T)+b[1]*np.sin(2*Pi*t/T)+b[2]*np.sin(3*Pi*t/T)\
+b[3]*np.sin(4*Pi*t/T)+b[4]*np.sin(5*Pi*t/T)+b[5]*np.sin(6*Pi*t/T)

y_3= 1/T*t+c[0]*np.sin(1*Pi*t/T)+c[1]*np.sin(2*Pi*t/T)+c[2]*np.sin(3*Pi*t/T)\
+c[3]*np.sin(4*Pi*t/T)+c[4]*np.sin(5*Pi*t/T)+c[5]*np.sin(6*Pi*t/T)


y_4= 1/T*t+d[0]*np.sin(1*Pi*t/T)+d[1]*np.sin(2*Pi*t/T)+d[2]*np.sin(3*Pi*t/T)\
+d[3]*np.sin(4*Pi*t/T)+d[4]*np.sin(5*Pi*t/T)+d[5]*np.sin(6*Pi*t/T)


y_5= 1/T*t+e[0]*np.sin(1*Pi*t/T)+e[1]*np.sin(2*Pi*t/T)+e[2]*np.sin(3*Pi*t/T)\
+e[3]*np.sin(4*Pi*t/T)+e[4]*np.sin(5*Pi*t/T)+e[5]*np.sin(6*Pi*t/T)


y = (1/T*t)**2


font1 = {'family' : 'Vera',
'weight' : 'normal',
'size'   : 12,
}
plt.figure(figsize=(6,4)) 

l1,=plt.plot(t,y_1,color="red",linewidth=2)#方程 
l2,= plt.plot(t,y_2,color="b",linewidth=2)
#l2, = plt.plot(t,y,color="grey",linewidth=2)
#plt.plot(t,y_1,color="g",linewidth=2)#方程 
#plt.plot(t,y_2,color="blue",linewidth=2)#方程 
l3,=plt.plot(t,y_3,color="grey",linewidth=2)#方程 
l4,=plt.plot(t,y_4,color="g",linewidth=2)
l5,=plt.plot(t,y_5,color="black",linewidth=2)
plt.legend(handles=[l1,l2,l3,l4,l5],labels=['5 qubits','6 qubits','7 qubits','9 qubits','11 qubits'],loc = 'best',ncol = 1,prop=font1)
plt.tick_params(labelsize=23)
#plt.xlabel("t",font1) 
#plt.ylabel("s",font1) 
#plt.title("10 bits RL path",font1) 
plt.ylim(-0.1,1)
plt.xlim(0,T) 
plt.savefig('10 bits RL path.pdf',bbox_inches='tight')
plt.show()
# fig=plt.figure()
# ax=fig.add_subplot(111)
# x=np.arange(-np.pi,np.pi,0.1)
# y=np.sin(x)
# z=np.cos(x)
# plt.plot(x,y,color="g",linewidth=2,label='sin')#方程 
# plt.plot(x,z,color="y",label='cos') 
# plt.xlabel("x") 
# plt.ylabel("y") 
# plt.title("sin,cos,hanshu") 
# plt.legend(loc='best')
# #plt.savefig('hanshu.png')

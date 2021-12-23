p1=np.array([1,1])
p2=np.array([4,4])
votes=[1,2,2,6,4,2,2,2,3]
points=np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p=np.array([2.5,2])
outcomes=np.array([0,0,0,0,1,1,1,1,1])



import matplotlib.pyplot as plt
plt.plot(points[:,0],points[:,1],"ro")
plt.plot(p[0],p[1],"bo")



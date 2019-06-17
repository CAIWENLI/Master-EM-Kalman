A=0.5;B=1;C=1;Q=2;R=4;
for j=1:10
    x(1)=randn(1,1);
    w=normrnd(0,Q,10,1);
    v=normrnd(0,R,9,1);
    for t=1:9
        x(t+1)=A*x(t)+B+w(t+1);
        y(t,j)=C*x(t)+v(t);
    end
end
for j=1:10
  logdata=y(:,j);
  [a(:,j),b(:,j),c(:,j),q(:,j),r(:,j),initx(:,j), V1(:,j),LL,xhat]=emkalman(logdata')
end
for n=1:10
    Ae(:,n)=(a(:,n)-A)*(a(:,n)-A)';
    Be(:,n)=(b(:,j)-B)*(b(:,j)-B)';
    Ce(:,n)=(c(:,j)-C)*(c(:,j)-C)';
    Qe(:,n)=(q(:,j)-Q)*(q(:,j)-Q)';
    Re(:,n)=(r(:,j)-R)*(r(:,j)-R)';
end   
Am=mean(Ae)
Bm=mean(Be)
Cm=mean(Ce)
Qm=mean(Qe)
Rm=mean(Re)


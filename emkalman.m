function [A, B, C, Q, R, initx, V1, LL,xhat] = emkalman(logdata)
%给定初值
%global y T 
for n=1:10
T=length(logdata);
A=rand(1,1);
B=rand(1,1);
C=rand(1,1);
Q=rand(1,1);
R=rand(1,1);
initx=logdata(1);
V1 = 0.1; 
y = logdata;
LL=[];
converged=0;
previous_loglik = -Inf;
max_iter = 500
num_iter = 0;
while ~converged &(num_iter <= max_iter)
xtt=zeros(1,T); Vtt=zeros(1,T); xtt1=zeros(1,T); Vtt1=zeros(1,T);
xtT=zeros(1,T); VtT=zeros(1,T); J=zeros(1,T); Vtt1T=zeros(1,T);
%应用Kalman滤波算法向前递推得到E[x(t)|y(1:t)]
x10=initx;
V10=V1;
for(t=1:T)
if(t==1)
xtt1(1) = initx; 
Vtt1(1) = V1;
else
xtt1(t) = A*xtt(t-1) + B;
Vtt1(t) = A*Vtt(t-1)*A' + Q;
end
Kt = Vtt1(t)*C/(C*Vtt1(t)*C'+R);
xtt(t) = xtt1(t) + Kt*(y(t) - C*xtt1(t));
Vtt(t) = Vtt1(t)-Kt*C*Vtt1(t);
end
KT = Kt;
%应用Kalman平滑从E[x(t)|y(1:t)]向后递推得到E[x(t)|y(1:T)]
xtT(T) = xtt(T);
VtT(T) = Vtt(T);
for(t=T:-1:2)
J(t-1) = Vtt(t-1)*A'/Vtt1(t);
xtT(t-1) = xtt(t-1)+ J(t-1)*(xtT(t)-(A*xtt(t-1)+B));
VtT(t-1) = Vtt(t-1)+ J(t-1)*(VtT(t)-Vtt1(t)).*J(t-1);
end
xhat = xtT %x_t的估计
Pt = VtT + xtT.*xtT; %E(x^2 | y]
%再次向后递推得到 E[x(t)x(t-1)|y(T)]
Vtt1T(T) = (1 - KT*C)*A*Vtt(T-1); %Var(x(T)x(T-1)|y(T))
for(t=T:-1:3)
Vtt1T(t-1) = Vtt(t-1)*J(t-2)'+J(t-1)*(Vtt1T(t)-A*Vtt(t-1))*J(t-2)';
end
Ptt1=[NaN Vtt1T(2:T)+xtT(2:T).*xtT(1:(T-1))]; 
end
%极大似然函数如下：
loglik=-sum((y-C*xhat).^2)/(2*R)-T*log(abs(R))/2-sum((xhat(2:T)-(A*xhat(1:(T-1))+B)).^2)/(2*Q)-(T-1)*log(abs(Q))/2-(xhat(1)-initx)^2/(2*V1)-log(abs(V1))/2-T*log(2*pi)/2;
if n==1
    loglik0=loglik;
    A0=A;
    B0=B;
    C0=C;
    Q0=Q;
    R0=R;
else
    if loglik0<loglik
    A0=A;
    B0=B;
    C0=C;
    Q0=Q;
    R0=R; 
    end   
end
end
A=A0;
B=B0;
C=C0;
Q=Q0;
R=R0;
%LL=[LL loglik];
%给定x_t的估计后，应用EM算法得到下列参数，不断进行更新A,B,C,Q,R,initx,initV1。直至上式收敛
A=sum(Ptt1(2:T)-B'*xtt1(t)')/sum(Pt(1:(T-1)));
B=(1/T)*sum(xhat(T)-A*xhat(T-1));
C=sum(y.*xhat)/sum(Pt(1:T));
R=(1/T)*sum(y.*y - C*xhat.*y-y.*xhat.*C+C*Pt(1:T).*C);
Q=(1/T)*sum(Pt(T)-A.*Ptt1(T)-Ptt1(T).*A-xhat.*B-B.*xhat+A.*Pt(T-1).*A+A.*xhat(T-1).*B+B.*xhat(T-1).*A+B.*B);
initx = xhat(1);
V1=Pt(1)-xhat(1)*xhat(1);
%检验收敛性
num_iter = num_iter+1;
converged = em_converged(loglik, previous_loglik);
previous_loglik = loglik;
end
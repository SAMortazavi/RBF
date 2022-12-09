clear
close all
clc
num_sample=100;
X=rand([num_sample 1]);
X=(sort(X))';
noise=(-0.5+rand([num_sample 1]))*0.1;
noise=noise';
y=sin(2*pi*X)+noise;


x=2*pi*X;
[mu,sig]=fit(x,y);
w21=random_Weight(3);


[w21,MSE]=train(x,y,mu,sig,w21);


test_x=(rand([num_sample 1]));
test_x=(sort(test_x))';
x_test=2*pi*test_x;


y_out_test=predict(x_test,w21,mu,sig);
y_out_real=sin(x_test);

figure()
plot(MSE)
figure()
subplot(1,2,1)
plot(y_out_real)
subtitle('real output')
subplot(1,2,2)
plot(y_out_test)
subtitle('predict output')



function [mu,sd]=fit(X,Y) %this function calculate mean and SD for data
    [idx,C]=kmeans([Y' X'],2);
    mu=[C(1,1) C(2,1)];
    sd=sqrt((C(1,1)-C(2,1))^2+(C(1,2)-C(2,2))^2);
    sd=sd/2/sqrt(2);
    return
end


function W21=update_Weight2(w21,y,yp,z,eta) %this function for updating last layer weights
W21=w21+eta*(y-yp)*yp*(1-yp)*z/3;
end


function W1=random_Weight(n) %this function for generating randomweights
W1=randn(1,n);
return
end


function out=my_gauss(x,mu,sig) %this function for generating gaussian output
    X=((x-mu)/sig)^2;
    x2=sqrt(2*pi)*sig;
    out=exp(-0.5*X)/x2;
end


function [w21,MSE]=train(x,y,mu,sig,w21) %this function trains weights
    eta=0.01;
    yp=[];
    E=15;
    MSE=[];
    epoch=0;
    while E>5
        if epoch>20000
            break
        end
        E=0;
        for i=1:100
            z1=my_gauss(x(i),mu(1),sig);
            z2=my_gauss(x(i),mu(2),sig);
            Z=[z1 z2 1];
            pre_out=dot(w21,Z);   
            yp(i)=pre_out;
            E=E+0.5*(y(i)-yp(i))^2;
            w21=update_Weight2(w21,y(i),yp(i),Z,eta);
        
        end
      epoch=epoch+1;
      MSE(epoch)=E;
      E
    end
    
end



function y_out=predict(x,w21,mu,sig) %this function predict output
            y_out=[]; 
            for i=1:100
                  z1=my_gauss(x(i),mu(1),sig);
                  z2=my_gauss(x(i),mu(2),sig);
                  Z=[z1 z2 1];
                  pre_out=dot(Z,w21);
                  y_out(i)=pre_out;
            end
return
end
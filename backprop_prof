
number = 10;
pi = 3.142;
learning_rate = 0.5;

[x_input] = [];
%x_input = rand(1,number);
for i = 1: number
[x_input] =  [x_input (pi / (2 * i) - 0.01)];
end

x_0 = [ 0 ];
x_input = [x_input,x_0];

[x_temp] = [];
%x_temp = rand(1,number);
i=number;
while i >= 1
[x_temp] =  [x_temp (-pi / (2 * i) + 0.01)];
i--;
end

x_input = [x_input,x_temp];

disp(x_input);



%x_input = sort(x_input);

required_output = rand(1,number*2 -1);
calculated_output = rand(1,number*2 -1);

x_hidden = [ 1 1 1 ];
x_output = [ 1 1 1 ];


y_first = [ 1 ];
y_second = [ 1 1 1 ];
y_third = [ 1 ];

delta_output = [ 1 ];
delta_hidden = [ 1 1 1 ];
delta_input = [ 1 ];

w_input = rand(1,1);
w_hidden = rand(1,3);
w_output = rand(1,3);

b_input = rand(1,1);
b_hidden = rand(1,3);
b_output = rand(1,1);



function [dy] = sigmoid_derivative(s)
dy = (4 * exp(-2*s)) ./ ((1 + exp(-2*s)) .* (1 + exp(-2*s)));
end

function [ out ] = sigmoidd( s )
out = (1 - exp(-2*s)) ./ (1 + exp(-2*s));
end

function [ out ] = forward_hidden( x , w , b )
j = 1;
for i = x
out(j)= w(j)*i + b(j);
% disp(out(j));
j= j + 1;

endfor;
end

function [ out ] = forward_output( x , w , b )
x_t = transpose(x);
out = w*x_t;
out = out + b;
end

function [ delta_hidden ] = propogate_delta_to_hidden(w_output, delta_output)
j = 1;  
for i = w_output
delta_hidden(j) = i * delta_output;
j = j+1;
endfor;
end

function [ delta_input ] = propogate_delta_to_input(w_hidden, delta_hidden)
w_hidden_t = transpose(w_hidden);
delta_input = delta_hidden*w_hidden_t;
end




for iter = 1  : 15
% Looping through all the input cases
count = 1;
%input = ceil(rand(1,1) * 21);
for i = x_input

x_hidden = [ 1 1 1 ];
x_output = [ 1 1 1 ];


y_first = [ 1 ];
y_second = [ 1 1 1 ];
y_third = [ 1 ];

delta_output = [ 1 ];
delta_hidden = [ 1 1 1 ];
delta_input = [ 1 ];
%
% Forward Pass begins:
%
%disp(i)
% layer 1
y_first = i * w_input(1) + b_input(1);
out = sigmoidd(y_first);
x_hidden = x_hidden * out;


y_second = forward_hidden(x_hidden,w_hidden,b_hidden);

% layer 2
k = 1;	
for j = y_second
x_output(k) = sigmoidd(j);
k = k + 1;	
endfor;

% layer 3
y_third = forward_output(x_output, w_output, b_output);
y_target = sigmoidd(y_third);
%disp(y_target);

% find actual tan answer
y_expected = tan(i);	
%disp(y_expected);

% calculating error
delta_output = y_expected - y_target;
error = delta_output;

%disp(error);

%delta_hidden = propogate_delta_to_hidden(w_output,delta_output);
%delta_input = propogate_delta_to_input(w_hidden,delta_hidden);

new_weight_hidden = [ 1 1 1];
new_weight_output = [1 1 1];
new_weight_input = [1];

new_weight_output = w_output - learning_rate.*error.*sigmoid_derivative(y_target).*x_output;

new_weight_hidden = w_hidden - learning_rate.*error.*sigmoid_derivative(y_second).*x_hidden.*w_output;

new_weight_input = w_input - (learning_rate*error*sigmoid_derivative(y_first)*i*w_hidden(1) + learning_rate*error*sigmoid_derivative(y_first)*i*w_hidden(2) + learning_rate*error*sigmoid_derivative(y_first)*i*w_hidden(3) );	

w_input = new_weight_input;
w_hidden = new_weight_hidden;
w_output = new_weight_output;

%updating bias

b_output = b_output - learning_rate.*error.*sigmoid_derivative(y_target);
b_hidden = b_hidden - learning_rate.*error.*sigmoid_derivative(y_second).*w_output;
b_input = b_input - (learning_rate*error*sigmoid_derivative(y_first)*w_hidden(1) + learning_rate*error*sigmoid_derivative(y_first)*w_hidden(2) + learning_rate*error*sigmoid_derivative(y_first)*w_hidden(3) );	


required_output(count) = tan(i);
calculated_output(count) = y_target;
count = count+1;

endfor;



% plot_x1 = x_input;
% plot_y1 = calculated_output;
% [val ind ] = sort(plot_x1);
% plot_y1 = plot_y1(ind);
%
% plot_x2 = sort(x_input);
% plot_y2 = tan(x_input);
% [val ind ] = sort(plot_x2);
% plot_y2 = plot_y2(ind);
% plot(plot_x1,plot_y1,plot_x2,plot_y2);

plot(x_input,required_output,x_input,calculated_output);
%axis([-pi/2,pi/2,-4,4])




endfor;

disp(required_output);
disp(calculated_output);
disp(w_input);
disp(w_hidden);
disp(w_output);

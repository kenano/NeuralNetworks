%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TEAM PROJECT - GROUP 3 
%  L - No of layers
%	n - No of neurons in input layer
%	h - No of neurons in hidden layer
%	m - No of neurons in output layer
%	u - Input vector
%	w - Weight vector
%	t - Target
%	y - output
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w] = backpropagation(no_of_iterations)
%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%
L = 3; 
n = 1;
h = 3;
m = 1;

%% Initialises random weight and bias as zero 
w = cell(L,1);
%% I/P layer
w{1} = [rand(n,1)] ;
%% Hidden Layer
w{2} = [rand(h,1)];
%%Output Layer
w{3} = [rand(h,1)];

b = cell(L,1);
b{1} = [zeros(n,1)];
b{2} = [zeros(h,1)];
b{3} = [zeros(m,1)];

x = 0+0.01:2*pi/200:(2*pi);
x = x(:);

t_fn = cos(x);

y = cell(L,1);
y{1} = [zeros(n,2)];
y{2} = [zeros(h,2)];
y{3} = [zeros(m,2)];

deltas = cell(L,1);
deltas{1} = [zeros(n,2)];
deltas{2} = [zeros(h,2)];
deltas{3} = [zeros(h,2)];

input_vector = [];
actual_op_vector =[];
output_vector = [];

for i=1:no_of_iterations
	%d = abs(ceil(rand() * 100 ))
	u = x(i);
	y = forwardneuron(u,w,b);
	t = t_fn(i);
	e = t-y{end}(end);
	deltas = calculate_delta(deltas,t,y,u,w);
	w = update_weights(deltas,t,y,w);
	input_vector = [input_vector u];
	actual_op_vector = [actual_op_vector t];
	output_vector = [output_vector y{end}(end)];
end

input_vector
actual_op_vector
output_vector
f = figure(1);
clf(f);
plot(input_vector,actual_op_vector,"r");
axis([0,2*pi,-1,1]);
hold on;
plot(input_vector,output_vector);

function [y] = forwardneuron(u,w,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To be replaced by code by Prateek Donni and Akshay Anil Kapoor
% 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v = w{1} * u + b{1};
y{1}(1) = v;   %vdash
y{1}(2) = sigmoid(v);    %v

z = [];
z = w{2} .* y{1}(2) + b{2};
y{2}(:,1) = z;			%zkdash
y{2}(:,2) = sigmoid(z);		%zk

ydash = w{3} .* y{2}(:,2) + b{3};
ydash = sum(ydash);
y{3}(1) = ydash;		%ydash
y{3}(2) = sigmoid(ydash);		%y
 

function [deltas] = calculate_delta(deltas,t,y,u,w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To be replaced by code by Sneha Singh and Kenan Ozdamar
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
deltas{3}(1) = sigmoid_derivative(y{3}(1));   %dy_db3
deltas{3}(:,2) = deltas{3}(1) .* y{2}(:,2);		%dy_dw3k

deltas{2}(:,1) = deltas{3}(1) * w{3} .* sigmoid_derivative(y{2}(:,1));   %dy_db2k
deltas{2}(:,2) = deltas{2}(:,1) * y{1}(2);			%dy_dw2k

dy_db1 = deltas{2}(:,1) .* w{2} .* sigmoid_derivative(y{1}(1));
deltas{1}(1) = sum(dy_db1);			%dy_db1
deltas{1}(2) = deltas{1}(1) * u;		%dy_dw1




function [w] = update_weights(deltas, t,y, w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code by Anusha Damodaran and Kiet Nyugen
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epsilon = 0.5;
e = t-y{end}(end);
L = 3;
for i = 1: L
change_in_weight = epsilon * e * deltas{i}(:,2);
w{i} = w{i} - (change_in_weight);
end

% Logistic Function G(s)
function [yd] = sigmoid(s)
yd = (1 - exp(-2*s)) ./ (1 + exp(-2*s));    %%Kenan

% Derivative of logistic function G'(s)
function [dy] = sigmoid_derivative(s)
dy = (4 * exp(-2*s)) ./ ((1 + exp(-2*s)) .* (1 + exp(-2*s))); 



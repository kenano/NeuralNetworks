

eta = 0.1;
number = 10;
pi = 3.142;

%[x_input] = [];
%x_input = rand(1,number);
%for i = 1: number
%[x_input] =  [x_input (-pi / (2 * i) )];
%end

%x_0 = [ 0 ];
%x_input = [x_input,x_0];

%[x_temp] = [];
%x_temp = rand(1,number);
%i=number;
%while i >= 1
%[x_temp] =  [x_temp (pi / (2 * i) )];
%i--;
%end


%x_input = [x_input,x_temp];

x_input = -pi/2:pi/20:pi/2;
disp(x_input);

size = 21;

input_vector = zeros(1,size);
required_output = zeros(1,size);
calculated_output = zeros(1,size);

x_hidden = [ 1 1 1 ];
x_output = [ 1 1 1 ];

y_first = [ 1 1 1 ];
y_second = [ 1 1 1 ];
y_third = [ 1 ];

w_input = rand(1,1);
w_hidden = rand(1,3);
w_output = rand(1,3);

b_input = rand(1,1);
b_hidden = rand(1,3);
b_output = rand(1,1);

delta_output = [ 1 ];
delta_hidden = [ 1 1 1 ];
delta_input = [ 1 ];

dw_input = [ 1 ];
dw_hidden = [ 1 1 1 ];
dw_output = [ 1 1 1 ];

prev_w_input = [ 1 ];
prev_w_hidden = [ 1 1 1 ];
prev_w_output = [ 1 1 1 ];

db_input = [ 1 ];
db_hidden = [ 1 1 1 ];
db_output = [ 1 ];



function [ out ] = sigmoidd( s )
out = (1 - exp(-2*s)) ./ (1 + exp(-2*s));
end


function [ out ] = sigmoidd_derivative( s )
out = (4 * exp(-2*s)) ./ ((1 + exp(-2*s)) .* (1 + exp(-2*s))) ;
end

function [ out ] = forward_hidden( x , w , b )
j = 1;
for i = x
out(j)= w(j)*i + b(j);
%disp(out(j));
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


for iter = 1 : 200
% Looping through all the input cases

% Looping through all the input cases
count = 1;
x = x_input;
n = rand(length(x),1);
[garbage index] = sort(n);
x_randomized = x(index);
for i = x_randomized
%
% Forward Pass begins:
%
delta_output = [ 1 ];
delta_hidden = [ 1 1 1 ];
delta_input = [ 1 ];

dw_input = [ 1 ];
dw_hidden = [ 1 1 1 ];
dw_output = [ 1 1 1 ];

x_hidden = [ 1 1 1 ];
x_output = [ 1 1 1 ];

y_first = [ 1 1 1 ];
y_second = [ 1 1 1 ];
y_third = [ 1 ];


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
y_expected = cos(i);	


% calculating error
delta_output = y_expected - y_target;


if(delta_output > 0.05 || delta_output < -0.05)

delta_hidden = propogate_delta_to_hidden(w_output,delta_output);
delta_input = propogate_delta_to_input(w_hidden,delta_hidden);


%updating weights

dw_input = eta * delta_input .* sigmoidd_derivative( y_first ) .* i;
prev_w_input = w_input;
w_input = w_input + dw_input;

dw_hidden = eta * delta_hidden .* sigmoidd_derivative( y_second ) .* x_hidden;
prev_w_hidden = w_hidden;
w_hidden = w_hidden + dw_hidden;

dw_output = eta * delta_output .* sigmoidd_derivative( y_third ) .* x_output;
prev_w_output = w_output;
w_output = w_output + dw_output;
 
%updating bias

db_input = eta * delta_input .* sigmoidd_derivative( y_first );
b_input = b_input + db_input;
 
db_hidden = eta * delta_hidden .* sigmoidd_derivative( y_second );
b_hidden = b_hidden + db_hidden;

db_output = eta * delta_output .* sigmoidd_derivative( y_third );
b_output = b_output + db_output;

 
end

input_vector(count) = i;
required_output(count) = cos(i);
calculated_output(count) = y_target;
count = count+1;



endfor;


%plot(input_vector,required_output,input_vector,calculated_output);
%axis([-pi/2,pi/2,-4,4]);



endfor;


 plot_x1 = input_vector;
 plot_y1 = calculated_output;
 [val ind ] = sort(plot_x1);
 plot_y1 = plot_y1(ind);

 plot_x2 = input_vector;
 plot_y2 = required_output;
 [val ind ] = sort(plot_x2);
 plot_y2 = plot_y2(ind);
 plot(sort(plot_x1),plot_y1,sort(plot_x2),plot_y2);
 
printf ("REquired output");
disp(required_output);
printf("Calculated output");
disp(calculated_output);
printf("WEights");
disp(w_input);
disp(w_hidden);
disp(w_output);










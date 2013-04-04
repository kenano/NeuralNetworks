% Neural network 1-3-1 learning the tangent curve
% This is the code to get C4 to generate w and b vector
%

total = 11;
number = (total-1)/2; 		% used to get random input values, 10 on the negative side, 10 on the positive side
pi = 3.1; 		% defining value of pi
eta = 0.1;  		% learning rate
limit = 0.25;		% limits of the range of inputs, eg. if limit is 0.5 i.e. 1/2, then the range is (-pi/2,pi/2)
iterations = 1000; 	% number of iterations
all_iter = [ 10 100 200 500 1000 ];
color = ["r" "g" "y" "b" "c"]
%
%
%
% Nomenclature for elements in the neural network:
%
%_______________________________________________________________________________________________
% Layer 1                   |                Layer 2            |          Layer 3            |
%_______________________________________________________________________________________________
%                           |                                   |                             |
%   bias = b_input	    |         bias = b_hidden           |     bias = b_output         |
%                           |                                   |                             |
%   input = x_input         |        input = x_hidden           |       input = x_output      |
%                           |                                   |                             |
%   weight = w_input        |         weight = w_hidden         |      weight = w_output      |
%                           |                                   |                             |
% for the summation function:|                                  |                             |
%   output = y_first        |       output = y_second           |       output = y_third      |
%                           |                                   |                             |
% for the output of tau function:                               |                             |
%   output = x_hidden       |         output = x_output         |       output = y_target     |
%                           |                                   |                             |
% Change in weights         |                                   |                             |
%     dw_input              |           dw_hidden               |           dw_output         |
%                           |                                   |                             |
% Previous iteration weights|                                   |                             | 
%   prev_w_input            |         prev_w_hidden             |         prev_w_output       |
%                           |                                   |                             |
% Change in bias            |                                   |                             |
%   db_input                |           db_hidden               |           db_output         |
%______________________________________________________________________________________________

%Defining the input vector
x_input = -pi*limit:pi*limit/number:pi*limit;
total = number*2+1;
scale=tan(max(x_input));

%The actual output that the neural network needs to learn
required_output = tan(sort(x_input));
calculated_output = [];

plot_x = sort(x_input);
plot_y1 = required_output;

%Initialising all variables

x_hidden = [ 1 1 1 ];
x_output = [ 1 1 1 ];

y_first = [ 1 1 1 ];
y_second = [ 1 1 1 ];
y_third = [ 1 ];

%w_input = 3*rand(1,1);
%w_hidden = 3*rand(1,3);
%w_output = 3*rand(1,3);

%seting values used for our iterations
w_input = [1.5505 ];
w_hidden = [2.4449   2.8375   2.2404];
w_output = [1.94873   0.50896   1.795 ];

%b_input = rand(1,1);
%b_hidden = rand(1,3);
%b_output = rand(1,1);

b_input = [0.89234];
b_hidden = [0.77330   0.85998   0.42386];
b_output = [ 0.62352 ];
 

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

%Sigmoid function G(s)
function [ out ] = sigmoidd( s )
	out = (1 - exp(-2*s)) ./ (1 + exp(-2*s));
end

%Sigmoid derivative function G'(s)
function [ out ] = sigmoidd_derivative( s )
	out = (4 * exp(-2*s)) ./ ((1 + exp(-2*s)) .* (1 + exp(-2*s))) ;
end


function [ delta_hidden ] = propogate_delta_to_hidden(w_output, delta_output)
	j = 1;	
	for i = w_output
		delta_hidden(j) = i * delta_output;
		j = j+1;
	endfor;
end

function[ D ] = calculateC3(Ynn,x_input,iterations)
    sum_of_error = 0;
    for loop = 1:iterations
        error = (Ynn(loop) - tan(x_input(loop)));
        error = (error^2);
        sum_of_error = sum_of_error + error;
    endfor;
    D =( 1/iterations ) * sqrt(sum_of_error);
    fprintf("\nD is %f",D);
end

function [ delta_input ] = propogate_delta_to_input(w_hidden, delta_hidden)
	w_hidden_t = transpose(w_hidden);
	delta_input = delta_hidden*w_hidden_t;
end

%
% The main Code starts here --------------------------------------------------------------
%
printf("Initial Weights\n");
disp(w_input);
disp(w_hidden);
disp(w_output);

printf("Initial Bias\n");
disp(b_input);
disp(b_hidden);
disp(b_output);

w_temp1 = w_input;
w_temp2 = w_hidden;
w_temp3 = w_output;

b_temp1 = b_input;
b_temp2 = b_hidden;
b_temp3 = b_output;

count = 1;

for(iterations = all_iter)
w_input = w_temp1;
w_hidden = w_temp2;
w_output = w_temp3;

b_input = b_temp1;
b_hidden = b_temp2;
b_output = b_temp3;



for iter = 1 : iterations
	% Looping through all the input cases
	% Select a random input from the input vector (u,y)
	ctr = ceil(rand(1,1) * total); 
	% Looping through all the input cases
	

	for i = x_input(ctr)
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


		% calculating error
		delta_output = y_expected - y_target;

		%Update weights only if error is greater than 0.05 or less than -0.05
		%if(delta_output > 0.05 || delta_output < 0.05)

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

		 
		%end

	endfor;

endfor;



% run forward passs again to get values
w(count,:) = [w_input w_hidden w_output];
b(count,:) = [b_input b_hidden b_output];

[calculated_output] = forward_pass( x_input , w(count,:) , b(count,:) );

%Plot the graph


hold on;
plot_y2 = calculated_output;
plot(plot_x,plot_y2,color(count));

%%Save the weights in matrix form
count = count + 1;

printf("Learned Weights for learning rate = %f\n",eta);
printf("Learned Weights for %d iterations\n",iterations);
printf("w_input = [");
printf("%f ",w_input);
printf("]\n");

printf("w_hidden = [");
printf("%f ",w_hidden);
printf("]\n");

printf("w_output = [");
printf("%f ",w_output);
printf("]\n");

printf("Learned Bias\n");
printf("b_input = [");
printf("%f ",b_input);
printf("]\n");

printf("b_hidden = [");
printf("%f ",b_hidden);
printf("]\n");

printf("b_output = [");
printf("%f ",b_output);
printf("]\n");

endfor;











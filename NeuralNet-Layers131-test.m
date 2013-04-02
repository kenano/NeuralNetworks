
number = 100;
pi = 3.1;
learning_rate = 0.1;
limit = 0.25;


%%%%%% input init %%%%%%%%%%%%%%%%%%%%%%%%%%

x_input = rand(1,number);						%creates a vector of size number, with element >0, <1 


x_input = x_input * limit *pi;					%normalize elements in vector so all values are < pi*limit (pi/4)

x_0 = [ 0 ];	
x_input = [x_input,x_0];						%add the elment 0.

x_temp = rand(1,number);						
x_temp = x_temp * limit * pi * (-1);			%same as above for negative values. 

x_input = [x_input,x_temp];						%combine the 2 parts to get 1 input set.

required_output = tan(sort(x_input));			%sort input and plug into tan function.


%plot(sort(x_input),required_output);			%plot of tan function, not processed through NN.

%%%%%Neural Net init %%%%%%%%%%%%%%%%%%%%%%%

layer2_input = [ 1 1 1 ];							%output of layer one put in a vector so it can be feed to layer 2.

v_prime = [ 1 ];									%stores layer1 linear combination output
v = [ 1 ];											%stores layer1  output

z_prime = [ 1 1 1 ];								%stores layer2 linear combination outputs
z = [ 1 1 1 ];										%stores layer2 outputs

y_prime = [ 1 ];									%stores layer3 linear combination outputs									
y = [1];											%stores layer3 output

w_input = rand(1,1)
w_hidden = rand(1,3)
w_output = rand(1,3)

b_input = rand(1,1)
b_hidden = rand(1,3)
b_output = rand(1,1)

y_expected = [1];

delta_weight_output = [ 1 ];
delta_weight_hidden = [ 1 1 1 ];
delta_weight_input = [ 1 ];

delta_bias_output = [ 1 ];
delta_bias_hidden = [ 1 1 1 ];
delta_bias_input = [ 1 ];


grad_output = [ 1 ];
grad_hidden = [ 1 1 1 ];

for iter = 1:100

	% Looping through all the input cases
	for input = x_input

		layer2_input = [ 1 1 1 ];

		z = [ 1 1 1 ];
		v = [ 1 ];
		y_prime = [ 1 ];

		grad_output = [ 1 ];
		grad_hidden = [ 1 1 1 ];
			
		delta_weight_output = [ 1 1 1];
		delta_weight_hidden = [ 1 1 1 ];
		delta_weight_input = [ 1 ];

		delta_bias_output = [ 1 1 1];
		delta_bias_hidden = [ 1 1 1 ];
		delta_bias_input = [ 1 ];	



		%%%%%%%%% Compute forward pass %%%%%%%%%%%%%%%%%%%%
		%%%%%%%%% Layer 1 %%%%%%%%%%%%%%%%%%%%


		v_prime = input * w_input(1) + b_input(1);			%first layer linear combination		
		v = sigmoidd( v_prime );							%first layer sigmoidd function
		layer2_input = layer2_input * v ;					%copy v (output from layer 1) to a vector so layer 2 can easily process.

		%%%%%%%%% Layer 2 %%%%%%%%%%%%%%%%%%%%
		j = 1;												%jth node in the second layer
		for i = layer2_input
			z_prime(j) = w_hidden(j) * i + b_hidden(j);
			j = j + 1;
		endfor;

		%z_prime

		j = 1;												%jth node in the second layer
		for i = z_prime
			z(j) = sigmoidd(i);	
			j = j + 1;							
		endfor;

		%z

		%%%%%%%%% Layer 3 %%%%%%%%%%%%%%%%%%%%
		z_t = transpose(z);									%not sure why I need to transpose.
		y_prime = w_output * z_t + b_output;
		y = sigmoidd(y_prime);


		%%%%%% test function %%%%%%%%%%%%%%%%%
		y_expected = tan(i);

		%%%%%%%% calculating gradient %%%%%%%%%%%
		grad_output(1) = (y_expected - y) * sigmoid_derivative(y);

		grad_hidden(1) = (1 - z(1)) * (z(1)) * grad_output(1) * (w_output(1));
		grad_hidden(2) = (1 - z(2)) * (z(2)) * grad_output(1) * (w_output(2));
		grad_hidden(3) = (1 - z(3)) * (z(3)) * grad_output(1) * (w_output(3));

		grad_input(1) = (1 - layer2_input(1)) * (layer2_input(1)) * (grad_hidden(1) * (w_hidden(1)) + grad_hidden(2) * (w_hidden(2)) + grad_hidden(3) * (w_hidden(3)));


		%%%%%%%%% calculating delta for weight %%%%%%%%%
		delta_weight_output(1) = learning_rate*grad_output(1) * z(1);
		delta_weight_output(2) = learning_rate*grad_output(1) * z(2);
		delta_weight_output(3) = learning_rate*grad_output(1) * z(3);

		delta_weight_hidden(1) = learning_rate*grad_hidden(1) * layer2_input(1);
		delta_weight_hidden(2) = learning_rate*grad_hidden(2) * layer2_input(2);
		delta_weight_hidden(3) = learning_rate*grad_hidden(3) * layer2_input(3);
				
		delta_weight_input(1) = learning_rate * grad_input(1) * input;

		%%%%%%%%%% calculating delta for bias %%%%%%%%%%%
		delta_bias_output(1) = learning_rate * grad_output(1);
		for temp = 1:3
			delta_bias_hidden(temp) = learning_rate * grad_hidden(temp);
		endfor;
		delta_bias_input(1) = learning_rate * grad_input(1);

		%%%%%%%%%%% updating weights %%%%%%%%%%%%%%%
		for temp = 1:3
			w_output(temp) = w_output(temp) + delta_weight_output(temp);
			w_hidden(temp) = w_hidden(temp) + delta_weight_hidden(temp);
		endfor;		
		w_input(1) = w_input(1) + delta_weight_input(1);

		%%%%%%%%%%% updating bias %%%%%%%%%%%%%%%%%%
		b_output(1) = b_output(1) + delta_bias_output(1);
		for temp = 1:3
			b_hidden(temp) = b_hidden(temp) + delta_bias_hidden(temp);
		endfor;		
		b_input(1) = b_input(1) + delta_bias_input(1);
	endfor;
endfor;

%%%%%%%%%%%%%% run forward passs again to get values %%%%%%%%%%%%%%
count = 1;
for input = sort(x_input)
	layer2_input = [ 1 1 1 ];

	z = [ 1 1 1 ];
	v = [ 1 ];
	y_prime = [ 1 ];

	%%%%%%%%% Compute forward pass %%%%%%%%%%%%%%%%%%%%
	%%%%%%%%% Layer 1 %%%%%%%%%%%%%%%%%%%%


	v_prime = input * w_input(1) + b_input(1);			%first layer linear combination		
	v = sigmoidd( v_prime );							%first layer sigmoidd function
	layer2_input = layer2_input * v ;					%copy v (output from layer 1) to a vector so layer 2 can easily process.

	%%%%%%%%% Layer 2 %%%%%%%%%%%%%%%%%%%%
	j = 1;												%jth node in the second layer
	for i = layer2_input
		z_prime(j) = w_hidden(j) * i + b_hidden(j);
		j = j + 1;
	endfor;

	%z_prime

	j = 1;												%jth node in the second layer
	for i = z_prime
		z(j) = sigmoidd(i);	
		j = j + 1;							
	endfor;

	%z

	%%%%%%%%% Layer 3 %%%%%%%%%%%%%%%%%%%%
	z_t = transpose(z);									%not sure why I need to transpose.
	y_prime = w_output * z_t + b_output;
	y = sigmoidd(y_prime);

	%%%%%%%%%%% create output vector %%%%%%%%
	calculated_output(count) = y;
	count = count + 1;

endfor;

calculated_output 

plot_x = sort(x_input);
plot(plot_x, calculated_output, plot_x, required_output);


%%%%%function definitions %%%%%%%%%%%%%%%%%%%%%%%

function [dy] = sigmoid_derivative(s)
dy = (4 * exp(-2*s)) ./ ((1 + exp(-2*s)) .* (1 + exp(-2*s))); 
end

function [ out ] = sigmoidd( s )
	out = (1 - exp(-2*s)) ./ (1 + exp(-2*s));
end


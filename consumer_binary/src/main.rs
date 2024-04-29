use neural_network::{activations::SIGMOID, matrix::Matrix, network::Network};
use std::{env, str::FromStr as StdFromStr};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Cli {
	#[structopt(short, long)]
	help: bool,
	#[structopt(short, long)]
	train: bool,
	#[structopt(short, long)]
	forward: bool,
	#[structopt(short, long)]
	looped_forward: bool,
	#[structopt(short, long,)]
	inputs: Option<String>,
}

fn split_inputs(s: String) -> Vec<Vec<f64>> {
	let mut inputs: Vec::<Vec<f64>> = Vec::<Vec<f64>>::new();
	let mut vector: Vec::<f64> = Vec::<f64>::new();
	let split = s.split(|c| c == ',' || c == ' ');	// split the string on the comma and the space
	let mut pass = 0;
	for i in split {
		if i.is_empty() {continue;}
		if pass == 0 {let _: f64 = match i.parse() {	// check if i is a number
			Ok(n) => {n}
			Err(e) => {println!("Error parsing number: {}", e); break;}};
			vector.push(f64::from_str(i).unwrap());
			pass += 1;
		} else {let _: f64 = match i.parse() {	// check if i is a number
			Ok(n) => {n}
			Err(e) => {println!("Error parsing number: {}", e); break;}};
			vector.push(f64::from_str(i).unwrap());
			pass = 0;
			inputs.push(vector.clone());
			// println!("{:?}", vector);	// Debugging
		vector.clear();}
	}
	// println!("{:?}", inputs);	// Debugging
return inputs;}

fn main() {
	env::set_var("RUST_BACKTRACE", "full");
	let args = Cli::from_args();
	println!("{:#?}", args);
	#[allow(unused)]	// Enables the compiler to know the variable 
	let mut inputs = vec![];	// exists when called from the if statements below

	if let Some(ref cli_inputs) = args.inputs {
		inputs = split_inputs(cli_inputs.clone());
	} else {
		inputs = vec![
			vec![0.0, 0.0],
			vec![0.0, 1.0],
 			vec![1.0, 0.0],
			vec![1.0, 1.0],
		];
	}

	let mut network = Network::new(vec![2,3,1], SIGMOID, 0.5);
	let targets = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];

	if args.train {
		training(&mut network, &inputs, &targets);
 		// network.train(inputs, targets, 10000);		
	}
	// test is the forward argument was passed
	if args.forward {
		if args.looped_forward {for _ in 0..100 {forward_pass(&mut network, &inputs);}
		} else {forward_pass(&mut network, &inputs);}
	}
	// test is the help argument was passed or if neither train or forward were passed
	if args.help || args.train == false && args.forward == false {
		usage(args);
		return;
	}

	fn usage(_args: Cli) {
		println!("Usage for {}:", env::args().next().unwrap_or_else(|| "program".to_string()));
		println!("	{} --train  <-- train the network", env::args().next().unwrap_or_else(|| "program".to_string()));
		println!("	{} --forward  <-- forward process the network", env::args().next().unwrap_or_else(|| "program".to_string()));
	}

	fn forward_pass(network: &mut Network, inputs: &[Vec<f64>]) {
		println!("Forward proccessing...");
		for input in inputs {
			print!("Input {:?}:  ", input);
			let result = network.feed_forward(Matrix::from(input.clone()));
			println!("{:?}", result);
		}
	}

	fn training(network: &mut Network, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) {
		let epochs = 100000;
		println!("Training {} epochs", epochs);
		network.train(inputs.clone(), targets.clone(), epochs);
	}
}
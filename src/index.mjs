import * as tf from '@tensorflow/tfjs-node';
import { getJsonFromCsv as parse } from 'convert-csv-to-json';

// Step 1: Prepare the Data
const data = parse('names.csv')

// Prepare the names
const correctNames = data.map(row => row['correct_name']);
const wrongNames = data.map(row => row['wrong_name']);

// Map the names to a 2D array
const inputData = wrongNames.map(name => [name.length]);
const outputData = correctNames.map(name => [name.length]);

// Define the input and outputs
const model = tf.sequential()
const inputTensor = tf.tensor2d(inputData);
const outputTensor = tf.tensor2d(outputData);

// Define the model
model.add(tf.layers.dense({ units: 32, inputShape: [1], activation: 'relu' }))
model.add(tf.layers.dense({ units: 1, activation: 'linear' }))
model.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['accuracy'] })

await model.fit(inputTensor, outputTensor, {
  epochs: 1000,
  callbacks: {
    onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}`)
  }
});

const inputName = 'Jn Smit';
const inputIndex = wrongNames.indexOf(inputName);
const correctedNameTensor = model.predict(tf.tensor2d([[inputName.length]]));
const correctedNameLength = correctedNameTensor.dataSync()[0];
const correctedName = inputName.slice(0, correctedNameLength) + correctNames[inputIndex].slice(correctedNameLength);

console.log(`Corrected name: ${correctedName}`);
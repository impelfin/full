onmessage = function (e) {
  let output = 0;
  let input = parseInt(e.data.input);
  let i = 2;
  while (i <= input) {
    if (input % i == 0) break;
    else i++;
  }
  output = input;
  output += input == i ? " is Prime Number." : " is not Prime Number.";
  postMessage(output);
};

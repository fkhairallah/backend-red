import { myAI } from "./model/langChainIngest";

console.log("Loading documents into Pinecone...");
let ai = new myAI();

async function foo() {
  await ai.init();
  if (!(await ai.checkPineconeIndex())) await ai.updatePinecone();
  //await ai.queryPineconeVectorStoreAndQueryLLM("how many engine does cerbo GX have?");
  console.log("Done.");
}

function doConsole() {
  // Get process.stdin as the standard input object.
  const standard_input = process.stdin;

  // Set input character encoding.
  standard_input.setEncoding("utf-8");

  // Prompt user to input data in console.
  console.log("Please input text in command line.");

  // When user input data and click enter key.
  standard_input.on("data", async function (question: string) {
    // User input exit.
    //console.log(`Got "${question}"`)
    
    if (question === "describe pinecone\r\n") {
      ai.describePinecone();
    } else
    if (question === "delete pinecone\r\n") {
      ai.deletePinecone();
    } else

    if(question === "reload pinecone\r\n") {
      ai.updatePinecone();

    } else
    if (question === "exit\r\n") {
      // Program exit.
      console.log("User input complete, program exit.");
      process.exit();
    } else {
      await ai.queryPineconeVectorStoreAndQueryLLM(question);
    }
  });
}

foo();
doConsole();

console.log("Main routine Exiting...");

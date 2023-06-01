import { PineconeClient } from "@pinecone-database/pinecone";
import { Document } from "langchain/document";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { DocxLoader} from "langchain/document_loaders/fs/docx"
import * as dotenv from "dotenv";

import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { LoadersMapping } from "langchain/dist/document_loaders/fs/directory";
import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain } from "langchain/chains"
import { Delete1Request, DescribeIndexStatsOperationRequest, DescribeIndexStatsRequest, QueryResponse } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";

export class myAI {
  indexName: string = "default";
  vectorDimension = 1536;
  docs: Document[] = [];
  pineClient: PineconeClient = new PineconeClient();

  constructor() {}

  async init() {
    console.log("Initialize AI class...");
    // init dotenv
    dotenv.config();


    // if index_name is not defined use 'default'
    this.indexName = process.env.PINECONE_INDEX_NAME ?? "default";


    // initialize Pinecone client
    this.pineClient = new PineconeClient();
    if (process.env.PINECONE_ENVIRONMENT && process.env.PINECONE_API_KEY)
      await this.pineClient.init({
        apiKey: process.env.PINECONE_API_KEY,
        environment: process.env.PINECONE_ENVIRONMENT,
      });
    else console.log("pinecone undefined");

    // initialize openAI

    console.log("initialization complete");
  }



  public async checkPineconeIndex():Promise<boolean> {
    // check existing index. If not found create it!
    console.log(`Checking "${this.indexName}"...`);

    const existingIndexes = await this.pineClient.listIndexes();

    if (!existingIndexes.includes(this.indexName)) {
      console.log(`Not found! Creating new....`);
      const createClient = await this.pineClient.createIndex({
        createRequest: {
          name: this.indexName,
          dimension: this.vectorDimension,
          metric: "cosine",
        },
      });

      console.log(createClient)

      // waste some time
      await new Promise((resolve) => setTimeout(resolve, 1000));
      



      console.log("Index created!");
      return false;
    }

    // we have an index
    return true;
  }

  public async updatePinecone() {
    console.log("retrieving pinecone index");

    const index = this.pineClient.Index(this.indexName);

    console.log("load PDF & TXT file into Pinecone")
    // define doc loaders
    const dl: LoadersMapping = {
      ".txt": (path: string) => new TextLoader(path),
      ".pdf": (path: string) => new PDFLoader(path),
      ".docx": (path: string) => new DocxLoader(path),
    };

    // load all documents into memory
    const directoryName = process.env.ASSETS_DIRECTORY ?? "./src/assets";
    console.log(`Loading files from ${directoryName}`)
    const loader = new DirectoryLoader(directoryName, dl);

    this.docs = await loader.load();


    console.log("Iterating:");

    let fileName = "";

    for (const doc of this.docs) {
      if (fileName != doc.metadata.source) {
        console.log(`Processing ${doc.metadata.source}`);
        fileName = doc.metadata.source;
      }
      const txtPath = doc.metadata.source;
      const text = doc.pageContent;
      const textsplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
      });
      const chunks = await textsplitter.createDocuments([text]);

      // now embed the text chunks
      const embeddingArrays = await new OpenAIEmbeddings().embedDocuments(
        chunks.map((chunk) => { return chunk.pageContent.replace(/\n/g, " ")})
      );

      // now create bachtes of vectors and insert them
      const batchSize = 100;
      let batch = [];
      for (let idx = 0; idx < chunks.length; idx++) {
        const chunk = chunks[idx];
        const vector = {
          id: `${txtPath}_${idx}`,
          values: embeddingArrays[idx],
          metadata: {
            ...chunk.metadata,
            loc: JSON.stringify(chunk.metadata.loc),
            pageContent: chunk.pageContent,
            txtPath: txtPath,
          },
        };
        batch.push(vector);

        // when batch is full or it's last item send to Pinecone
        if (batch.length === batchSize || idx === chunks.length - 1) {
          await index.upsert({
            upsertRequest: {
              vectors: batch,
            },
          });
          batch = [];
        }
      }
    }
  }

  public async describePinecone() {
    const pIndex = this.pineClient.Index(this.indexName);
     const describeIndexStatsQuery: DescribeIndexStatsRequest = {
       filter: {},
     };
     const foo: DescribeIndexStatsOperationRequest = {
       describeIndexStatsRequest: describeIndexStatsQuery
     };
    const stats = await pIndex.describeIndexStats(foo);
    
    console.log(stats);
  }

  public async deletePinecone() {
        const pIndex = this.pineClient.Index(this.indexName);
        const ri:Delete1Request = {
          deleteAll: true
        };
        const stats = await pIndex.delete1(ri);

        console.log(stats);

  }



  public async queryPineconeVectorStoreAndQueryLLM(question: string) {

    //console.log("Query");

    const pIndex = this.pineClient.Index(this.indexName);

    //create query embedding
    const queryEmbedding = await new OpenAIEmbeddings().embedQuery(question);

    // get top 10 matched from pinecone
    let queryResponse:QueryResponse = await pIndex.query({
      queryRequest: {
        topK: 10,
        vector: queryEmbedding,
        includeMetadata: true,
        includeValues: true,
      }
    });

    //console.log(`Pinecone found ${queryResponse.matches?.length} matches...`);

    if (queryResponse.matches?.length) {

      //console.log("Querying openAI...")
      const llm = new OpenAI();
      const chain = loadQAStuffChain(llm);

      //console.log("Concat..")
      // extract and concatenance pages
      let concatenatedPageContent = (queryResponse.matches?.map((match:any) => 
        {return match.metadata.pageContent}).join(" ")) ?? " "

      //console.log(`calling chain with ${concatenatedPageContent.length/4} tokens`)

      //concatenatedPageContent = "42 is the answer to life the universe and everything"
      // execute the chain with input

      const result = await chain.call({
        input_documents: [ new Document({pageContent: concatenatedPageContent})],
        question: question,
      })

      console.log(`Answer: ${result.text}\nQuestion:`)


    }
    else
    {
      console.warn("No matches. No ask.")
    }

   }
}

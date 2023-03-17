import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { pinecone } from '@/utils/pinecone-client';
import { PDFLoader } from 'langchain/document_loaders';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import fs from 'fs';
import path from 'path';
import * as cliProgress from 'cli-progress';

/* Name of directory to retrieve files from. You can change this as required */
const filePath = 'docs/MorseVsFrederick.pdf';

export const run = async (pdfFilesPaths: string[]) => {
  // Create a new progress bar instance and customize its appearance
  const progressBar = new cliProgress.SingleBar({
    format: 'Progress |' + '{bar}' + '| {percentage}% || {value}/{total} items',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true,
  });

  const embedProgressBar = new cliProgress.SingleBar({
    format:
      'Embed Progress |' + '{bar}' + '| {percentage}% || {value}/{total} items',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true,
  });

  // Start the progress bar
  progressBar.start(pdfFilesPaths.length, 0);

  try {
    for (const filePath of pdfFilesPaths) {
      progressBar.increment();
      console.log('Processing file: ', filePath);

      /*load raw docs from the pdf file in the directory */
      const loader = new PDFLoader(filePath);
      // const loader = new PDFLoader(filePath);
      const rawDocs = await loader.load();

      // console.log(rawDocs);

      /* Split text into chunks */
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      const docs = await textSplitter.splitDocuments(rawDocs);
      // console.log('split docs', docs);

      console.log('creating vector store...');
      /*create and store the embeddings in the vectorStore*/
      const embeddings = new OpenAIEmbeddings();
      const index = pinecone.Index(PINECONE_INDEX_NAME); //change to your own index name

      //embed the PDF documents

      /* Pinecone recommends a limit of 100 vectors per upsert request to avoid errors*/
      const chunkSize = 50;

      // Start the progress bar
      embedProgressBar.start((docs.length - 1) / chunkSize, 0);
      for (let i = 0; i < docs.length; i += chunkSize) {
        embedProgressBar.increment();
        const chunk = docs.slice(i, i + chunkSize);
        // console.log('chunk', i, chunk);
        await PineconeStore.fromDocuments(
          index,
          chunk,
          embeddings,
          'text',
          PINECONE_NAME_SPACE,
        );
      }
      embedProgressBar.stop();
    }
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to ingest your data');
  }

  progressBar.stop();
};

const getPdfFilePaths = (dir: string, fileList: string[] = []): string[] => {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      fileList = getPdfFilePaths(filePath, fileList);
    } else if (path.extname(file).toLowerCase() === '.pdf') {
      fileList.push(filePath);
    }
  }

  return fileList;
};

(async () => {
  const directoryPath = './docs';
  const pdfFilesPaths = getPdfFilePaths(directoryPath);
  await run(pdfFilesPaths);
  console.log('ingestion complete');
})();

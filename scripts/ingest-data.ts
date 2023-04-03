import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import {PineconeStore, SupabaseVectorStore} from 'langchain/vectorstores';
import { CustomPDFLoader } from '@/utils/customPDFLoader';
import { DirectoryLoader } from 'langchain/document_loaders';
import type { SupabaseClient } from '@supabase/supabase-js';
import {supabaseClient} from "@/utils/supabase-client";
import fs from "fs/promises";

/* Name of directory to retrieve your files from */
const filePath = 'docs';

export const run = async () => {
    try {
        /*load raw docs from the all files in the directory */
        const directoryLoader = new DirectoryLoader(filePath, {
            '.pdf': (path) => new CustomPDFLoader(path),
        });

        // const loader = new PDFLoader(filePath);
        const rawDocs = await directoryLoader.load();

        /* Split text into chunks */
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const docs = await textSplitter.splitDocuments(rawDocs);
        // console.log(docs);
        console.log(`${rawDocs.length} documents detected`);
        
        const json = JSON.stringify(rawDocs);
        await fs.writeFile('leyton-hr.json', json);
        console.log('json file containing data saved on disk');

        /*create and store the embeddings in the vectorStore*/
        console.log('creating vector store...');

        //embed the PDF documents
        let errors = await supabaseClient.from('documents_hr').delete().neq('id', 0);
        
        await new SupabaseVectorStore(
          supabaseClient,
          new OpenAIEmbeddings(),
          {
              tableName: 'documents_hr',
              queryName: 'match_documents_hr'
          }
        ).addDocuments(docs);

    } catch (error) {
        console.log('error', error);
        throw new Error('Failed to ingest your data');
    }
};

(async () => {
    await run();
    console.log('ingestion complete');
})();

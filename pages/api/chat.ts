import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { SupabaseVectorStore } from 'langchain/vectorstores';
import { openai } from '@/utils/openai-client';
import { supabaseClient } from '@/utils/supabase-client';
import { makeChain } from '@/utils/makechain';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const { question, history, countryCode, target } = req.body;

  // let country = countries.filter(function(c) { return c.code === countryCode; })[0]

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');
  
  let tableName: string,
      queryName: string;
  
  switch (target) {
    case 'hr':
      tableName = 'documents_hr'
      queryName = 'match_documents_hr'
      break;
    default : // website
      tableName = 'documents'
      queryName = 'match_documents'
      break;
  }

  const vectorStore = await new SupabaseVectorStore(
    supabaseClient,
    new OpenAIEmbeddings(),
    {
      tableName: tableName,
      queryName: queryName
    }
  );

  // console.log('queryName', queryName)
  // console.log('tableName', tableName)
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
  });

  const sendData = (data: string) => {
    res.write(`data: ${data}\n\n`);
  };

  sendData(JSON.stringify({ data: '' }));

  const model = openai;
  // create the chain
  const chain = makeChain(vectorStore, (token: string) => {
    sendData(JSON.stringify({ data: token }));
  });

  try {
    //Ask a question
    const response = await chain.call({
      question: sanitizedQuestion,
      chat_history: history || [],
    });

    // console.log('----------------------------------------------')
    // console.log('sanitizedQuestion', sanitizedQuestion);
    // console.log('----------------------------------------------')
    // console.log('history', history);
    // console.log('----------------------------------------------')
    // console.log('response', response);
    // console.log('----------------------------------------------')
  } catch (error) {
    console.log('error', error);
  } finally {
    sendData('[DONE]');
    res.end();
  }
}

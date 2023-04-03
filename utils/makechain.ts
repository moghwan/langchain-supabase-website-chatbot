import { OpenAI } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { SupabaseVectorStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a professional question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are a helpful AI assistant called LeytonGPT, representing an international consulting firm called Leyton.
  If the question is in another language you can translate all your answers to the same language, including your introduction.
You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided, do not make up anything.
The answer should be short at maximum of three lines unless you're asked to explain more about it.
If you can't find the answer in the context below, just say "Hmm, maybe I'm missing something. Can you give more context?" Don't try to make up an answer.
If the question is not related to Leyton or the context provided, politely inform them that you are tuned to only answer questions that are related to Leyton.

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: SupabaseVectorStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0.4 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAI({
      modelName: 'text-davinci-003',
      temperature: 0.4,
      streaming: Boolean(onTokenStream),
      callbackManager: {
        handleNewToken: onTokenStream,
      },
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
  });
};

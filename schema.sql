-- Enable the pgvector extension to work with embedding vectors
create extension vector;

-- Create a table to store your documents
create table documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Create a function to search for documents
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where similarity >= 0.8
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create an index to be used by the search function
create index on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- hr documents
-- Create a table to store your documents_hr
create table documents_hr (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Create a function to search for documents_hr
create or replace function match_documents_hr (
  query_embedding vector(1536),
  match_count int
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents_hr.embedding <=> query_embedding) as similarity
  from documents_hr
  where similarity >= 0.8
  order by documents_hr.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create an index to be used by the search function
create index on documents_hr
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
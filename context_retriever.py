from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import SQLDatabase, VectorStoreIndex

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
import re

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core  import ServiceContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from sqlalchemy import text
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext
import os
from pathlib import Path
from typing import Dict
from typing import List

class ContextRetriever():
    def __init__(self):
        self.engine = create_engine("sqlite:////content/worlddb.db")

        self.sql_database = SQLDatabase(engine)

        self.table_schema_objs = [
            SQLTableSchema(table_name="city", context_str="City"),
            SQLTableSchema(table_name="country", context_str="country"),
            SQLTableSchema(table_name="countrylanguage", context_str="countrylanguage")
        ]

        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        self.table_index_dir = "table_index_dir"

        if not Path(self.table_index_dir).exists():
            os.makedirs(self.table_index_dir)

        self.vector_index_dict = {}
        self.engine = self.sql_database.engine
        for table_name in self.sql_database.get_usable_table_names():
            print(f"Indexing rows in table: {table_name}")
            if not os.path.exists(f"{self.table_index_dir}/{table_name}"):
                # get all rows from table
                with self.engine.connect() as conn:
                    cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                    result = cursor.fetchall()
                    row_tups = []
                    for row in result:
                        row_tups.append(tuple(row))

                # index each row, put into vector store index
                nodes = [TextNode(text=str(t)) for t in row_tups]

                # put into vector store index (use OpenAIEmbeddings by default)
                index = VectorStoreIndex(nodes, embed_model=self.embed_model)

                # save index
                index.set_index_id("vector_index")
                index.storage_context.persist(f"{self.table_index_dir}/{table_name}")
            else:
                # rebuild storage context
                storage_context = StorageContext.from_defaults(
                    persist_dir=f"{self.table_index_dir}/{table_name}"
                )
                # load index
                index = load_index_from_storage(
                    storage_context, index_id="vector_index", embed_model=self.embed_model
                )
            self.vector_index_dict[table_name] = index

    def get_table_context_and_rows_str(
        query_str: str
    ):
        """Get table context string."""
        context_strs = []
        for table_schema_obj in self.table_schema_objs:
            # first append table info + additional context
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            # also lookup vector index to return relevant table rows
            vector_retriever = self.vector_index_dict[
                table_schema_obj.table_name
            ].as_retriever(similarity_top_k=2)
            relevant_nodes = vector_retriever.retrieve(query_str)
            if len(relevant_nodes) > 0:
                table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                table_info += table_row_context

            context_strs.append(table_info)
        return "\n\n".join(context_strs)

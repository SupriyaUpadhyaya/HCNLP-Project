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
        self.engine = create_engine("sqlite:////content/drive/MyDrive/HCNLP-Text2Sql-Project/worlddb.db")

        self.sql_database = SQLDatabase(self.engine)

        self.table_schema_objs = [
            SQLTableSchema(table_name="city", context_str="Stores information about cities. It has attributes like name (Name), country code (CountryCode), district (District), and population (Population). CountryCode is a foreign key referencing the Code field in the country table, indicating the country the city belongs to."),
            SQLTableSchema(table_name="country", context_str="Stores information about countries. Each country has a unique three-letter code (Code) as the primary key. It holds details like country name (Name), continent (Continent), region (Region), land area (SurfaceArea), independence year (IndepYear) if applicable, population (Population), life expectancy (LifeExpectancy), and various economic data (GNP, GNPOld). It also stores information on government form (GovernmentForm), head of state (HeadOfState), and capital city reference (Capital - likely referencing the ID from the city table). Additional details include local name (LocalName), a secondary country code (Code2), and official language information (linked to the countrylanguage table)."),
            SQLTableSchema(table_name="countrylanguage", context_str="Connects countries with their spoken languages. A combination of CountryCode (referencing the country table) and Language (language name) forms the primary key, ensuring a unique relationship between a country and its languages. It has an IsOfficial flag indicating if the language is officially recognized in the country. It also stores the percentage (Percentage) of the population speaking that language.")
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

    def get_table_context_and_rows_str(self,
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

refiner_template = """
### Instruction:
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
### Constraints:
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
### Query:
-- {query}
### Evidence:
{evidence}
### Database info:
{desc_str}
### Foreign keys:
{fk_str}
### old SQL:
```sql
{sql}
```
### SQLite error: 
{sqlite_error}
### Exception class:
{exception_class}

Now please fixup old SQL and generate new SQL again.
### correct SQL:
"""

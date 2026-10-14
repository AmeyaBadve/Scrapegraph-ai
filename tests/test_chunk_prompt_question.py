import pytest
from langchain_core.prompts import PromptTemplate
from scrapegraphai import prompts


@pytest.mark.parametrize(
    "template_name",
    ["TEMPLATE_CHUNKS_MD", "TEMPLATE_CHUNKS"],
)
def test_chunk_prompts_include_user_question(template_name):
    """Ensure each chunk receives the user's extraction question."""
    from langchain_core.prompts import PromptTemplate
    from scrapegraphai import prompts

    template = getattr(prompts, template_name)

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={
            "content": "Django and Flask are Python web frameworks.",
            "chunk_id": 1,
            "format_instructions": "Return valid JSON.",
        },
    )

    rendered = prompt.format(question="Which Python frameworks are mentioned?")

    assert "USER QUESTION: Which Python frameworks are mentioned?" in rendered
    assert "Django and Flask" in rendered

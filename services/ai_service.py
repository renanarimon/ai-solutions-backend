import logging

from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_community.llms.fake import FakeListLLM
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# 5 technical documents about NVIDIA internal R&D tools and CUDA optimization
TECHNICAL_DOCUMENTS = [
    "NVIDIA Nsight Systems is an internal R&D profiling tool that provides a system-wide view of application performance. "
    "It helps identify CPU throttling, GPU idle time, and kernel launch gaps. Use it to trace CUDA API calls and correlate "
    "CPU and GPU timelines for optimizing throughput in multi-GPU and heterogeneous workloads.",
    "NVIDIA Nsight Compute is a kernel-level profiler for CUDA applications. It offers detailed metrics on warp execution, "
    "memory throughput, and occupancy. R&D teams use it to pinpoint bottlenecks in GPU kernels and compare against "
    "theoretical peak performance. Key features include the Roofline model and source-level correlation.",
    "CUDA optimization tip: Coalesced global memory access is critical for performance. Ensure threads in a warp access "
    "consecutive memory locations so that the GPU can merge requests into a single transaction. Use shared memory to "
    "batch and reorder non-coalesced accesses when necessary.",
    "NVIDIA internal R&D relies on the CUDA Graph API to reduce launch overhead in repetitive workloads. Capture sequences "
    "of kernels and memcpy operations once, then replay with a single host call. This minimizes latency in inference "
    "pipelines and iterative solvers where the same operations run thousands of times.",
    "Optimizing for Tensor Cores: Structure matrix multiplies to use FP16 or BF16 with FP32 accumulation where supported. "
    "Keep dimensions multiples of 8 for FP16 and 16 for INT8. NVIDIA internal tools validate tensor core utilization "
    "and flag kernels that fall back to CUDA Cores, which is essential for maximum throughput on Ampere and Hopper GPUs.",
]


class AIService:
    """Encapsulates LangChain LLM, memory, retriever, and chain for the AI Solutions API."""

    def __init__(self, initial_messages: list[str] | None = None):
        self.db: list[str] = list(initial_messages or [])

        self._llm = FakeListLLM(
            responses=[
                "That's an interesting point. I'd suggest exploring it further.",
                "Thanks for sharing. Here's a concise summary: you're on the right track.",
                "Got it. One way to look at it is to break it down into smaller steps.",
                "I understand. Consider trying a different approach and see what works.",
            ]
        )

        self._memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="text",
        )
        for user_text in self.db:
            self._memory.chat_memory.add_user_message(user_text)
            self._memory.chat_memory.add_ai_message("(processed)")

        self._retriever = self._setup_retriever()

        self._prompt = PromptTemplate(
            input_variables=["chat_history", "text", "context"],
            template=(
                "Relevant context from the knowledge base:\n{context}\n\n"
                "Previous conversation:\n{chat_history}\n\n"
                "User said: {text}\n\nAssistant:"
            ),
        )
        self._chain = LLMChain(
            llm=self._llm,
            prompt=self._prompt,
            memory=self._memory,
        )

    def _setup_retriever(self):
        """Initialize the vector store with technical documents and return a retriever."""
        embeddings = DeterministicFakeEmbedding(size=1536)
        documents = [Document(page_content=text) for text in TECHNICAL_DOCUMENTS]
        vector_store = DocArrayInMemorySearch.from_documents(documents, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 1})

    def process(self, text: str) -> tuple[str, int, list[str]]:
        """
        Process user text: retrieve relevant context, run the chain with context, return (answer, num_tokens, source_documents).
        """
        self.db.append(text)

        # Retrieve the most relevant document for the user's query
        retrieved_docs = self._retriever.invoke(text)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else "No relevant context found."
        source_documents = [doc.page_content for doc in retrieved_docs]

        # Debug: print retrieved context to the terminal
        logger.info("Retrieved context (debug):\n%s", context)

        result = self._chain.invoke({"text": text, "context": context})
        answer = result["text"]

        memory_content = self._memory.load_memory_variables({})
        logger.info(
            "Current memory content:\n%s",
            memory_content.get("chat_history", "(empty)"),
        )

        num_tokens = len(text.split())
        return answer, num_tokens, source_documents

    def get_messages(self) -> list[str]:
        """Return all messages stored via process."""
        return self.db

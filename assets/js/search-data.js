// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-bookshelf",
          title: "bookshelf",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/books/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-readings",
          title: "Readings",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/readings/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-understanding-transformers-the-architecture-behind-modern-llms",
        
          title: "Understanding Transformers — The Architecture Behind Modern LLMs",
        
        description: "A deep dive into the Transformer architecture — self-attention, multi-head attention, position encodings, and scaling laws.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/formatting-and-links/";
          
        },
      },{id: "books-beej-39-s-guide-to-c-programming",
          title: 'Beej&amp;#39;s Guide to C Programming',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/beej/";
            },},{id: "books-cuda-by-example",
          title: 'Cuda by example',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/cuda/";
            },},{id: "books-designing-distributed-systems",
          title: 'Designing Distributed Systems',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/distributed/";
            },},{id: "books-programming-massively-parallel-processors",
          title: 'Programming Massively Parallel Processors',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/parallel/";
            },},{id: "books-python-concurrency-with-asyncio",
          title: 'Python Concurrency with asyncio',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/python/";
            },},{id: "books-reinforcement-learning-an-introduction",
          title: 'Reinforcement Learning - An Introduction',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/rl/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "readings-beej-39-s-guide-to-c-programming",
          title: 'Beej&amp;#39;s Guide to C Programming',
          description: "HTML simple E-book version of Beej&#39;s Guide to C Programming.",
          section: "Readings",handler: () => {
              window.location.href = "/readings/rd0/";
            },},{id: "readings-how-to-think-about-gpus-a-special-section-of-39-how-to-scale-your-model-39",
          title: 'How to think about GPUs - A Special section of &amp;#39;How to Scale...',
          description: "Part 12 of How To Scale Your Model (Part 11: Conclusion | The End)",
          section: "Readings",handler: () => {
              window.location.href = "/readings/rd1/";
            },},{id: "readings-nvidia-tensor-core-evolution-from-volta-to-blackwell",
          title: 'NVIDIA Tensor Core Evolution: From Volta To Blackwell',
          description: "Technical overview of the evolution of NVIDIA&#39;s Tensor Core architecture, from the Volta GPU to the latest Blackwell GPU, highlighting the key architectural changes, performance improvements, and programming model advancements over the generations.",
          section: "Readings",handler: () => {
              window.location.href = "/readings/rd2/";
            },},{id: "readings-how-to-optimize-a-cuda-matmul-kernel-for-cublas-like-performance-a-worklog",
          title: 'How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog',
          description: "Iterative optimization of a CUDA matrix multiplication (SGEMM) kernel to achieve near-peak performance on a GPU, exploring techniques such as global memory coalescing, shared memory caching, and increasing arithmetic intensity.",
          section: "Readings",handler: () => {
              window.location.href = "/readings/rd3/";
            },},{id: "readings-llm-training-on-gpu-clusters-ultra-scale-playbook",
          title: 'LLM Training on GPU Clusters: Ultra-Scale Playbook',
          description: "An ultra-scale playbook of LLM Training on GPU Clusters containing the theory, the code and efficiency benchmarking.",
          section: "Readings",handler: () => {
              window.location.href = "/readings/rd4/";
            },},{id: "readings-how-to-scale-your-model",
          title: 'How to Scale your model',
          description: "A Systems view of LLMs on TPUs (and GPUs) - A playbook",
          section: "Readings",handler: () => {
              window.location.href = "/readings/rd5/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6E%61%72%69%6E%65%30%32%33%33@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/Narayan-21", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/nryn-221", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://narayan-21.github.io/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];

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
        },{id: "nav-publications",
          title: "publications",
          description: "Coming soon",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
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
      },{id: "books-cuda-by-example",
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

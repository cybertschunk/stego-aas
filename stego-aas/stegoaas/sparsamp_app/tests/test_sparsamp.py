from django.test import TestCase

from ..encoding import full_encode
from ..decoding import full_decode
from ..model_manager import get_model_manager

class SparSampTest(TestCase):

    def test_short_text(self):
        context = "Once upon a time"
        text = "attack@dawn"
        seed = 12345
        messages = full_encode(context, text, seed)
        decoded_text, attempts_list = full_decode(context, messages, seed)
        print(f"BackCheck attempts: {attempts_list}, Total: {sum(attempts_list)}")
        self.assertEqual(decoded_text, text)

    def test_multiple_texts(self):
        context= "Steganography is the practice of hiding messages within other non-secret text or data."
        texts = [
            "JavaScript is a versatile programming language essential for modern web development. Originally created by Brendan Eich in 1995 JavaScript has evolved from simple browser scripting to full-stack development. It enables interactive web pages through DOM manipulation event handling and asynchronous programming. Node.js extended JavaScript to server-side development creating a unified language ecosystem. Popular frameworks like React Angular and Vue.js simplify complex application development. JavaScript supports functional and object-oriented programming paradigms with modern features like arrow functions promises and async/await. Its ubiquity means JavaScript runs on billions of devices worldwide. Package managers like npm provide access to vast libraries accelerating development significantly.",

            "Git is a distributed version control system created by Linus Torvalds for Linux kernel development. It tracks changes in source code during software development enabling collaboration among programmers. Unlike centralized systems Git gives every developer a complete repository copy with full history. Branching and merging capabilities allow parallel development without conflicts. GitHub GitLab and Bitbucket provide cloud hosting for Git repositories adding collaboration features. Common commands include commit push pull merge and rebase for managing code versions. Git enables experimentation through branches while maintaining stable main codebases. Its distributed nature ensures code backup and availability. Modern software development relies heavily on Git for coordinating team efforts efficiently.",

            "Docker revolutionized software deployment through containerization technology making applications portable across environments. Containers package applications with dependencies ensuring consistent behavior regardless of hosting platform. Unlike virtual machines Docker containers share the host operating system kernel reducing overhead significantly. Docker images define container blueprints while registries like Docker Hub store and distribute them. Orchestration tools like Kubernetes manage container clusters at scale. Microservices architecture benefits greatly from Docker enabling independent service deployment and scaling. DevOps practices embrace Docker for continuous integration and deployment pipelines. Container isolation improves security by limiting application access to system resources. Docker simplifies development by eliminating environment configuration issues between development and production systems.",
        ]
        seed = 23456789
        for i in range(0, len(texts)):
            text = texts[i]
            print(f"Running test {i+1} of {len(texts)}")
            messages = full_encode(context, text[:200], seed)
            decoded_text, attempts_list = full_decode(context, messages, seed)
            print(f"  BackCheck attempts: {attempts_list}, Total: {sum(attempts_list)}")
            self.assertEqual(decoded_text, text[:200])


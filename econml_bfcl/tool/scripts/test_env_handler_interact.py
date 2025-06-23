#!/usr/bin/env python3
"""
EnvHandler 完整功能测试脚本

测试所有核心功能：
1. 空消息列表处理
2. 工具调用执行
3. 多轮对话流程
4. 错误处理
5. 对话完成检测
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from verl.verl.tools.env_handler import EnvHandler, env_step


def create_test_entry():
    """创建测试条目"""
    return {
        "id": "multi_turn_miss_func_0",
        "question": [
            [
                {
                    "role": "user",
                    "content": "Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "Perform a detailed search using grep to identify sections in the file pertaining to 'budget analysis'.",
                }
            ],
            [
                {
                    "role": "user",
                    "content": "Upon identifying the requisite 'budget analysis' content, sort the 'final_report.pdf' by line for improved clarity and comprehension.",
                }
            ],
            [],
            [
                {
                    "role": "user",
                    "content": "Move 'previous_report.pdf' in document directory to temp as well and having final report also there, proceed to juxtapose it with 'previous_report.pdf' to detect any critical alterations.",
                }
            ],
        ],
        "function": [
            {
                "name": "authenticate_twitter",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Authenticate a user with username and password.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "Username of the user.",
                        },
                        "password": {
                            "type": "string",
                            "description": "Password of the user.",
                        },
                    },
                    "required": ["username", "password"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "authentication_status": {
                            "type": "boolean",
                            "description": "True if authenticated, False otherwise.",
                        }
                    },
                },
            },
            {
                "name": "comment",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Comment on a tweet for the authenticated user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "tweet_id": {
                            "type": "integer",
                            "description": "ID of the tweet to comment on.",
                        },
                        "comment_content": {
                            "type": "string",
                            "description": "Content of the comment.",
                        },
                    },
                    "required": ["tweet_id", "comment_content"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "comment_status": {
                            "type": "string",
                            "description": "Status of the comment action.",
                        }
                    },
                },
            },
            {
                "name": "follow_user",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Follow a user for the authenticated user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "username_to_follow": {
                            "type": "string",
                            "description": "Username of the user to follow.",
                        }
                    },
                    "required": ["username_to_follow"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "follow_status": {
                            "type": "boolean",
                            "description": "True if followed, False if already following.",
                        }
                    },
                },
            },
            {
                "name": "get_tweet",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retrieve a specific tweet.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "tweet_id": {
                            "type": "integer",
                            "description": "ID of the tweet to retrieve.",
                        }
                    },
                    "required": ["tweet_id"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "ID of the retrieved tweet.",
                        },
                        "username": {
                            "type": "string",
                            "description": "Username of the tweet's author.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the tweet.",
                        },
                        "tags": {
                            "type": "array",
                            "description": "List of tags associated with the tweet.",
                            "items": {"type": "string"},
                        },
                        "mentions": {
                            "type": "array",
                            "description": "List of users mentioned in the tweet.",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            {
                "name": "get_tweet_comments",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retrieve all comments for a specific tweet.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "tweet_id": {
                            "type": "integer",
                            "description": "ID of the tweet to retrieve comments for.",
                        }
                    },
                    "required": ["tweet_id"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "comments": {
                            "type": "array",
                            "description": "List of dictionaries, each containing comment information.",
                            "items": {
                                "type": "dict",
                                "properties": {
                                    "username": {
                                        "type": "string",
                                        "description": "Username of the commenter.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of the comment.",
                                    },
                                },
                            },
                        }
                    },
                },
            },
            {
                "name": "get_user_stats",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Get statistics for a specific user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "Username of the user to get statistics for.",
                        }
                    },
                    "required": ["username"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "tweet_count": {
                            "type": "integer",
                            "description": "Number of tweets posted by the user.",
                        },
                        "following_count": {
                            "type": "integer",
                            "description": "Number of users the specified user is following.",
                        },
                        "retweet_count": {
                            "type": "integer",
                            "description": "Number of retweets made by the user.",
                        },
                    },
                },
            },
            {
                "name": "get_user_tweets",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retrieve all tweets from a specific user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "Username of the user whose tweets to retrieve.",
                        }
                    },
                    "required": ["username"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "user_tweets": {
                            "type": "array",
                            "description": "List of dictionaries, each containing tweet information.",
                            "items": {
                                "type": "dict",
                                "properties": {
                                    "id": {
                                        "type": "integer",
                                        "description": "ID of the retrieved tweet.",
                                    },
                                    "username": {
                                        "type": "string",
                                        "description": "Username of the tweet's author.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of the tweet.",
                                    },
                                    "tags": {
                                        "type": "array",
                                        "description": "List of tags associated with the tweet.",
                                        "items": {"type": "string"},
                                    },
                                    "mentions": {
                                        "type": "array",
                                        "description": "List of users mentioned in the tweet.",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        }
                    },
                },
            },
            {
                "name": "list_all_following",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: List all users that the authenticated user is following.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
                "response": {
                    "type": "dict",
                    "properties": {
                        "following_list": {
                            "type": "array",
                            "description": "List of all users that the authenticated user is following.",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
            {
                "name": "mention",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Mention specified users in a tweet.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "tweet_id": {
                            "type": "integer",
                            "description": "ID of the tweet where users are mentioned.",
                        },
                        "mentioned_usernames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of usernames to be mentioned.",
                        },
                    },
                    "required": ["tweet_id", "mentioned_usernames"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "mention_status": {
                            "type": "string",
                            "description": "Status of the mention action.",
                        }
                    },
                },
            },
            {
                "name": "post_tweet",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Post a tweet for the authenticated user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content of the tweet.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tags for the tweet. Tag name should start with #. This is only relevant if the user wants to add tags to the tweet.",
                            "default": [],
                        },
                        "mentions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of users mentioned in the tweet. Mention name should start with @. This is only relevant if the user wants to add mentions to the tweet.",
                            "default": [],
                        },
                    },
                    "required": ["content"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "ID of the posted tweet.",
                        },
                        "username": {
                            "type": "string",
                            "description": "Username of the poster.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content of the tweet.",
                        },
                        "tags": {
                            "type": "array",
                            "description": "List of tags associated with the tweet.",
                            "items": {"type": "string"},
                        },
                        "mentions": {
                            "type": "array",
                            "description": "List of users mentioned in the tweet.",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            {
                "name": "posting_get_login_status",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Get the login status of the current user.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
                "response": {
                    "type": "dict",
                    "properties": {
                        "login_status": {
                            "type": "boolean",
                            "description": "True if the current user is logged in, False otherwise.",
                        }
                    },
                },
            },
            {
                "name": "retweet",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retweet a tweet for the authenticated user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "tweet_id": {
                            "type": "integer",
                            "description": "ID of the tweet to retweet.",
                        }
                    },
                    "required": ["tweet_id"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "retweet_status": {
                            "type": "string",
                            "description": "Status of the retweet action.",
                        }
                    },
                },
            },
            {
                "name": "search_tweets",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Search for tweets containing a specific keyword.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword to search for in the content of the tweets.",
                        }
                    },
                    "required": ["keyword"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "matching_tweets": {
                            "type": "array",
                            "description": "List of dictionaries, each containing tweet information.",
                            "items": {
                                "type": "dict",
                                "properties": {
                                    "id": {
                                        "type": "integer",
                                        "description": "ID of the retrieved tweet.",
                                    },
                                    "username": {
                                        "type": "string",
                                        "description": "Username of the tweet's author.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of the tweet.",
                                    },
                                    "tags": {
                                        "type": "array",
                                        "description": "List of tags associated with the tweet.",
                                        "items": {"type": "string"},
                                    },
                                    "mentions": {
                                        "type": "array",
                                        "description": "List of users mentioned in the tweet.",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        }
                    },
                },
            },
            {
                "name": "unfollow_user",
                "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Unfollow a user for the authenticated user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "username_to_unfollow": {
                            "type": "string",
                            "description": "Username of the user to unfollow.",
                        }
                    },
                    "required": ["username_to_unfollow"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "unfollow_status": {
                            "type": "boolean",
                            "description": "True if unfollowed, False if not following.",
                        }
                    },
                },
            },
            {
                "name": "cat",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file from current directory to display. No path is allowed. ",
                        }
                    },
                    "required": ["file_name"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "file_content": {
                            "type": "string",
                            "description": "The content of the file.",
                        }
                    },
                },
            },
            {
                "name": "cd",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Change the current working directory to the specified folder.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": "The folder of the directory to change to. You can only change one folder at a time. ",
                        }
                    },
                    "required": ["folder"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "current_working_directory": {
                            "type": "string",
                            "description": "The new current working directory path.",
                        }
                    },
                },
            },
            {
                "name": "cp",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Copy a file or directory from one location to another.  If the destination is a directory, the source file or directory will be copied into the destination directory.  Both source and destination must be local to the current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "The name of the file or directory to copy.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The destination name to copy the file or directory to. If the destination is a directory, the source will be copied into this directory. No file paths allowed. ",
                        },
                    },
                    "required": ["source", "destination"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the copy operation or an error message if the operation fails.",
                        }
                    },
                },
            },
            {
                "name": "diff",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Compare two files of any extension line by line at the current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name1": {
                            "type": "string",
                            "description": "The name of the first file in current directory.",
                        },
                        "file_name2": {
                            "type": "string",
                            "description": "The name of the second file in current directorry. ",
                        },
                    },
                    "required": ["file_name1", "file_name2"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "diff_lines": {
                            "type": "string",
                            "description": "The differences between the two files.",
                        }
                    },
                },
            },
            {
                "name": "du",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Estimate the disk usage of a directory and its contents.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "human_readable": {
                            "type": "boolean",
                            "description": "If True, returns the size in human-readable format (e.g., KB, MB). ",
                            "default": False,
                        }
                    },
                    "required": [],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "disk_usage": {
                            "type": "string",
                            "description": "The estimated disk usage.",
                        }
                    },
                },
            },
            {
                "name": "echo",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Write content to a file at current directory or display it in the terminal.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to write or display.",
                        },
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file at current directory to write the content to. Defaults to None. ",
                            "default": "None",
                        },
                    },
                    "required": ["content"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "terminal_output": {
                            "type": "string",
                            "description": "The content if no file name is provided, or None if written to file.",
                        }
                    },
                },
            },
            {
                "name": "find",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Find any file or directories under specific path that contain name in its file name.  This method searches for files of any extension and directories within a specified path that match the given name. If no name is provided, it returns all files and directories in the specified path and its subdirectories. Note: This method performs a recursive search through all subdirectories of the given path.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": 'The directory path to start the search. Defaults to the current directory (".").',
                            "default": ".",
                        },
                        "name": {
                            "type": "string",
                            "description": "The name of the file or directory to search for. If None, all items are returned. ",
                            "default": "None",
                        },
                    },
                    "required": [],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "matches": {
                            "type": "array",
                            "description": "A list of matching file and directory paths relative to the given path.",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
            {
                "name": "grep",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Search for lines in a file of any extension at current directory that contain the specified pattern.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file to search. No path is allowed and you can only perform on file at local directory.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "The pattern to search for. ",
                        },
                    },
                    "required": ["file_name", "pattern"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "matching_lines": {
                            "type": "array",
                            "description": "Lines that match the pattern.",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
            {
                "name": "ls",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: List the contents of the current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {
                            "type": "boolean",
                            "description": "Show hidden files and directories. Defaults to False. ",
                            "default": False,
                        }
                    },
                    "required": [],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "current_directory_content": {
                            "type": "array",
                            "description": "A list of the contents of the specified directory.",
                            "items": {"type": "string"},
                        }
                    },
                },
            },
            {
                "name": "mkdir",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Create a new directory in the current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "dir_name": {
                            "type": "string",
                            "description": "The name of the new directory at current directory. You can only create directory at current directory.",
                        }
                    },
                    "required": ["dir_name"],
                },
                "response": {"type": "dict", "properties": {}},
            },
            {
                "name": "mv",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Move a file or directory from one location to another. so",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source name of the file or directory to move. Source must be local to the current directory.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something, destination is the new file name. ",
                        },
                    },
                    "required": ["source", "destination"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the move operation.",
                        }
                    },
                },
            },
            {
                "name": "pwd",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Return the current working directory path.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
                "response": {
                    "type": "dict",
                    "properties": {
                        "current_working_directory": {
                            "type": "string",
                            "description": "The current working directory path.",
                        }
                    },
                },
            },
            {
                "name": "rm",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Remove a file or directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file or directory to remove. ",
                        }
                    },
                    "required": ["file_name"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the remove operation.",
                        }
                    },
                },
            },
            {
                "name": "rmdir",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Remove a directory at current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "dir_name": {
                            "type": "string",
                            "description": "The name of the directory to remove. Directory must be local to the current directory. ",
                        }
                    },
                    "required": ["dir_name"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the remove operation.",
                        }
                    },
                },
            },
            {
                "name": "tail",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the last part of a file of any extension.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file to display. No path is allowed and you can only perform on file at local directory.",
                        },
                        "lines": {
                            "type": "integer",
                            "description": "The number of lines to display from the end of the file. Defaults to 10. ",
                            "default": 10,
                        },
                    },
                    "required": ["file_name"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "last_lines": {
                            "type": "string",
                            "description": "The last part of the file.",
                        }
                    },
                },
            },
            {
                "name": "touch",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Create a new file of any extension in the current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the new file in the current directory. file_name is local to the current directory and does not allow path.",
                        }
                    },
                    "required": ["file_name"],
                },
                "response": {"type": "dict", "properties": {}},
            },
            {
                "name": "wc",
                "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Count the number of lines, words, and characters in a file of any extension from current directory.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Name of the file of current directory to perform wc operation on.",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Mode of operation ('l' for lines, 'w' for words, 'c' for characters). ",
                            "default": "l",
                        },
                    },
                    "required": ["file_name"],
                },
                "response": {
                    "type": "dict",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "The count of the number of lines, words, or characters in the file.",
                        },
                        "type": {
                            "type": "string",
                            "description": 'The type of unit we are counting. [Enum]: ["lines", "words", "characters"]',
                        },
                    },
                },
            },
        ],
        "involved_classes": ["TwitterAPI", "GorillaFileSystem"],
        "missed_function": {
            "3": [
                {
                    "name": "sort",
                    "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Sort the contents of a file line by line.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "description": "The name of the file appeared at current directory to sort. ",
                            }
                        },
                        "required": ["file_name"],
                    },
                    "response": {
                        "type": "dict",
                        "properties": {
                            "sorted_content": {
                                "type": "string",
                                "description": "The sorted content of the file.",
                            }
                        },
                    },
                }
            ]
        },
        "path": [
            "GorillaFileSystem.find",
            "GorillaFileSystem.mv",
            "GorillaFileSystem.grep",
            "GorillaFileSystem.sort",
            "GorillaFileSystem.diff",
            "TwitterAPI.post_tweet",
        ],
    }


def test_empty_messages():
    """测试空消息列表"""
    print("=== 测试1: 空消息列表 ===")
    
    env_handler = EnvHandler("test_env")
    test_entry = create_test_entry()
    
    response = env_handler.interact([], test_entry)
    print(response)
    assert response["messages"][-1]["role"] == "user"
    assert "Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory" in response["messages"][-1]["content"]
    
    print("✅ 空消息列表测试通过")
    return response


def test_tool_calls():
    """测试工具调用"""
    print("\n=== 测试2: 工具调用 ===")
    
    env_handler = EnvHandler("test_env")
    test_entry = create_test_entry()
    
    # 模拟包含工具调用的assistant消息
    messages = [
        {"role": "user", "content": "Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mkdir",
                        "arguments": '{"dir_name": "backup"}'
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "mv",
                        "arguments": '{"source": "report.txt", "destination": "backup"}'
                    }
                }
            ]
        }
    ]
    
    response = env_handler.interact(messages, test_entry)
    print(response)
    assert response["message"]["role"] == "tool"
    
    print("✅ 工具调用测试通过")
    return response


def test_multi_turn_conversation():
    """测试完整的多轮对话"""
    print("\n=== 测试3: 多轮对话 ===")
    
    env_handler = EnvHandler("test_env")
    test_entry = create_test_entry()
    
    messages = []
    turn_count = 0
    max_turns = 10  # 防止无限循环
    
    while turn_count < max_turns:
        # 获取环境响应
        response = env_handler.interact(messages, test_entry)
        
        # 检查是否对话结束
        if response["messages"][-1]["content"] == "[CONVERSATION_COMPLETED]":
            print(f"对话在第 {turn_count} 轮完成")
            break
        
        # 添加环境响应到消息历史
        env_message = response["messages"]
        messages.append(env_message)
        
        print(f"轮次 {turn_count}: {env_message[-1]['role']} - {env_message[-1]['content'][:50]}...")
        
        # 模拟policy响应
        if env_message[-1]["role"] == "user":
            # 对用户消息，模拟assistant响应（可能包含工具调用）
            if turn_count == 0:
                # 第一轮：文件操作
                policy_response = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{turn_count}_1",
                            "type": "function",
                            "function": {
                                "name": "mkdir",
                                "arguments": '{"dir_name": "backup"}'
                            }
                        }
                    ]
                }
            elif turn_count == 2:
                # 第二轮：搜索操作
                policy_response = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{turn_count}_1",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": '{"pattern": "预算", "file_name": "report.txt"}'
                            }
                        }
                    ]
                }
            elif turn_count == 4:
                # 第三轮：发推文
                policy_response = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{turn_count}_1",
                            "type": "function",
                            "function": {
                                "name": "post_tweet",
                                "arguments": '{"content": "测试推文内容"}'
                            }
                        }
                    ]
                }
            else:
                # 其他情况，普通回复
                policy_response = {
                    "role": "assistant",
                    "content": f"我已经处理了第 {turn_count//2 + 1} 轮的请求。"
                }
        else:
            # 对工具响应，模拟简单的assistant回复
            policy_response = {
                "role": "assistant",
                "content": "操作已完成。"
            }
        
        messages.append(policy_response)
        turn_count += 1
    
    print("✅ 多轮对话测试通过")
    return messages


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试4: 错误处理 ===")
    
    env_handler = EnvHandler("test_env")
    test_entry = create_test_entry()
    
    # 测试1: 最后一条消息不是assistant
    messages = [{"role": "user", "content": "Hello"}]
    response = env_handler.interact(messages, test_entry)
    assert "[ERROR]" in response["messages"][-1]["content"]
    print("✅ 错误消息格式测试通过")
    
    # 测试2: 无效的工具调用格式
    messages = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "tool_calls": [{"invalid": "format"}]
        }
    ]
    response = env_handler.interact(messages, test_entry)
    # 应该能处理无效格式而不崩溃
    print("✅ 无效工具调用格式测试通过")
    
    # 测试3: 超出轮次范围
    messages = [
        {"role": "user", "content": "test1"},
        {"role": "assistant", "content": "response1"},
        {"role": "user", "content": "test2"},
        {"role": "assistant", "content": "response2"},
        {"role": "user", "content": "test3"},
        {"role": "assistant", "content": "response3"},
        {"role": "user", "content": "test4"},
        {"role": "assistant", "content": "final response"},
        {"role": "user", "content": "test5"},
        {"role": "assistant", "content": "final response"},
        {"role": "user", "content": "test6"},
        {"role": "assistant", "content": "final response"}
    ]
    response = env_handler.interact(messages, test_entry)
    print(response)
    assert response["messages"][-1]["content"] == "[CONVERSATION_COMPLETED]"
    print("✅ 对话完成检测测试通过")


def test_simplified_interface():
    """测试简化接口"""
    print("\n=== 测试5: 简化接口 ===")
    
    test_entry = create_test_entry()
    
    # 测试简化接口
    response = env_step([], test_entry)
    assert response["messages"][-1]["role"] == "user"
    
    print("✅ 简化接口测试通过")


def test_conversation_completion():
    """测试对话完成检测"""
    print("\n=== 测试6: 对话完成检测 ===")
    
    env_handler = EnvHandler("test_env")
    test_entry = create_test_entry()
    
    # 模拟已经完成所有轮次的消息历史
    messages = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "回复1"},
        {"role": "user", "content": "第二轮"},
        {"role": "assistant", "content": "回复2"},
        {"role": "user", "content": "第三轮"},
        {"role": "assistant", "content": "最终回复"},
        {"role": "user", "content": "第四轮"},
        {"role": "assistant", "content": "最终回复"},
        {"role": "user", "content": "第五轮"},
        {"role": "assistant", "content": "最终回复"}
    ]
    
    response = env_handler.interact(messages, test_entry)
    assert response["messages"][-1]["content"] == "[CONVERSATION_COMPLETED]"
    
    print("✅ 对话完成检测测试通过")


def main():
    """运行所有测试"""
    print("开始 EnvHandler 完整功能测试\n")
    
    try:
        test_empty_messages()
        # test_tool_calls()
        test_multi_turn_conversation()
        test_error_handling()
        test_simplified_interface()
        test_conversation_completion()
        
        print("\n🎉 所有测试通过！EnvHandler 功能正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
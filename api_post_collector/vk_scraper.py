import argparse
import json
from vk_group import VKGroup


def main(access_token, group_domain, output_file, post_count):
    vk_group = VKGroup(access_token, group_domain)

    dataset = vk_group.get_posts(offset=0, count=post_count)

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(dataset, json_file, ensure_ascii=False, indent=4)

    print(f"Data saved to file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for fetching post data from a VK group.")
    parser.add_argument("--access_token", type=str,
                        required=True, help="VK API access token.")
    parser.add_argument("--group_domain", type=str,
                        required=True, help="VK group domain.")
    parser.add_argument("--output_file", type=str,
                        default="vk_posts.json", help="File name for saving data.")
    parser.add_argument("--post_count", type=int, required=True,
                        help="Number of posts to fetch.")

    args = parser.parse_args()
    main(args.access_token, args.group_domain,
         args.output_file, args.post_count)

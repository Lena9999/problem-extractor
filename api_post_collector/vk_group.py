from tqdm import tqdm
import requests
from datetime import datetime


class VKGroup:
    """
    A class for collecting wall posts from a VKontakte group using the API.

    Functions:
    - get_posts: Fetches posts from the group's wall, returns their IDs and text.
    - post_count: Returns the total number of posts in the group.
    """

    def __init__(self, access_token, group_domain, api_version="5.131"):
        self.access_token = access_token
        self.group_domain = group_domain
        self.api_version = api_version
        self.base_url = "https://api.vk.com./method/"

      # Check the validity of the token during initialization
        error_message = self._is_token_valid()
        if error_message:
            raise ValueError(f"Access token error: {error_message}")

    def _make_request(self, method, params):
        """Internal function to send requests to the VK API"""
        url = self.base_url + method
        request_params = {
            'access_token': self.access_token,
            'v': self.api_version,
        }
        request_params.update(params)
        response = requests.get(url, params=request_params)
        return response.json()

    def _is_token_valid(self):
        """Method to check if the token is valid"""
        try:
            # Make a simple request to the API (for example, users.get)
            response = self._make_request('users.get', {})
            if 'error' in response:
                return response['error']['error_msg']
            return None
        except requests.RequestException as e:
            return f"Network error: {e}"

    def get_posts(self, count=1, offset=0):
        """Method to fetch and return both the ID and text of posts from the wall"""
        all_posts = []
        # the maximum number of posts per request (VK API limitation)"
        batch_size = 100

        # Create a progress bar with the total number of posts to fetch
        with tqdm(total=count, desc="Downloading posts", unit="post") as pbar:
            while len(all_posts) < count:
                params = {
                    'domain': self.group_domain,
                    'count': min(batch_size, count - len(all_posts)),
                    'offset': offset
                }

                data = self._make_request('wall.get', params)

                if 'error' in data:
                    raise Exception(f"Error fetching posts: {
                                    data['error']['error_msg']}")

                posts = data.get('response', {}).get('items', [])

                if not posts:
                    break

                all_posts.extend([
                    (
                        post.get('id'),
                        post.get('text', ''),
                        datetime.fromtimestamp(post.get('date')).strftime(
                            '%Y-%m-%d %H:%M:%S')
                    )
                    for post in posts
                ])

                offset += len(posts)

                pbar.update(len(posts))

        return all_posts[:count]

    def post_count(self):
        """Returns the total number of posts in the group."""
        params = {
            'domain': self.group_domain,
            'count': 1
        }

        data = self._make_request('wall.get', params)

        if 'error' in data:
            raise Exception(f"Error fetching total posts count: {
                            data['error']['error_msg']}")

        total_count = data.get('response', {}).get('count', 0)

        return total_count

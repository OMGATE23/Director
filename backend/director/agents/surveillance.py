import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session, 
    MsgStatus, 
    TextContent, 
    VideoContent, 
    SearchResultsContent,
    SearchData,
    VideoData,
    ShotData
)
from director.tools.videodb_tool import VideoDBTool
from videodb import SceneExtractionType, IndexType

logger = logging.getLogger(__name__)


SCENE_INDEX_PROMPT= """
    The given video is supposed to be of Survellience Footage

    For the analysis consider the following points.

    Describe the scene with the following:
    People: Describe the people, their arrival, their probable profession and expressions
    Actions: Outline any activities or movements occurring such as delivery driver arriving, people leaving etc.
    Time: Describe the time of the scene. If the time is available at display do display it aswell
    Suspicious Activities: Identify any burglar, stalker etc.
    Other: Identify any animals, birds etc which is visible in the scene

""".strip()

DEFAULT_SCENE_INDEX_CONFIG = {
    "type": SceneExtractionType.shot_based,
    "config": {
        "threshold": 20,
        "frame_count": 5,
        "min_scene_len": 5,
    },
}

SURVEILLANCE_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "video_id": {
            "type": "string",
            "description": "The ID of the video to process",
        },
        "collection_id": {
            "type": "string",
            "description": "The ID of the collection to process",
        },
        "query": {
            "type": "string", 
            "description": "Search query"
        },
        "scene_index_prompt": {
            "type": "string",
            "description": "The prompt to use for scene indexing",
        },
        "index_threshold": {
            "type": "number",
            "description": "Sensitivity of the model towards scene changes within the video",
        },
        "index_min_scene_len": {
            "type": "number",
            "description": "Minimum length of a scene in frames",
        },
        "index_frame_count": {
            "type": "number",
            "description": "Number of frames to extract per scene",
        },
        "result_threshold": {
            "type": "number",
            "description": " Initial filter for top N matching documents",
        },
        "score_threshold": {
            "type": "integer",
            "description": "Absolute threshold filter for relevance scores",
        },
        "dynamic_score_percentage": {
            "type": "integer",
            "description": "Adaptive filtering mechanism: Useful when there is a significant gap between top results and tail results after score_threshold filter. Retains top x% of the score range. Calculation: dynamic_threshold = max_score - (range * dynamic_score_percentage)",
        }
    },
    "required": ["query", "video_id", "collection_id"],
}


class SurveillanceAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "surveillance"
        self.description = "Agent to help you with your surveillance footage"
        self.parameters = SURVEILLANCE_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
            self, 
            collection_id: str, 
            video_id: str, 
            query: str,
            index_threshold=20,
            index_min_scene_len=15,
            index_frame_count=4,
            scene_index_prompt=SCENE_INDEX_PROMPT,
            result_threshold=5,
            score_threshold=0.2,
            dynamic_score_percentage=20,
        ) -> AgentResponse:
        """
        Search for a particular event or occurence and we will take you right to the point
        We will index the scenes in the particular video. After this, the user can ask for various questions for which you need to answer or/and show a stream of videos

        This agent is used to keep an eye on important events such as arrival of delivery driver, intruders arriving, animals entering the house etc.

        :param str collection_id: The ID of the collection where the video is present.
        :param video_id: The id of the video for which the video player is required.
        :param query: the search query.
        :return: The response containing information about the search results, compiled video and text summary of the event shown in the video.
        :rtype: AgentResponse
        """
        try:
            self.output_message.actions.append("Started Processing Surveillance Footage")

            search_result_content = SearchResultsContent(
                status=MsgStatus.progress,
                status_message="Started searching in the surveilleance footage",
                agent_name=self.agent_name,
            )

            self.output_message.content.append(search_result_content)
            self.output_message.push_update()

            videodb_tool = VideoDBTool(collection_id=collection_id)
            video = videodb_tool.get_video(video_id=video_id)
            scene_index_id = None

            try:
                scene_id = videodb_tool.index_scene(
                    video_id=video_id,
                    extraction_type=SceneExtractionType.shot_based,
                    extraction_config={
                        "threshold": index_threshold,
                        "min_scene_len": index_min_scene_len,
                        "frame_count": index_frame_count,
                    },
                    prompt=scene_index_prompt
                )
                scene_index_id = videodb_tool.get_scene_index(video_id=video_id, scene_id=scene_id)
            except:
                scene_index_list = videodb_tool.list_scene_index(video_id)
                if scene_index_list:
                    scene_index_id = scene_index_list[0].get("scene_index_id")

            
            if not scene_index_id:
                self.output_message.actions.append("Scene index not found")
                self.output_message.push_update()
                raise ValueError("Scene indexing returned no results")

            search_results = videodb_tool.semantic_search(
                query,
                index_type=IndexType.scene,
                video_id=video_id,
                result_threshold=result_threshold,
                dynamic_score_percentage=dynamic_score_percentage,
                score_threshold=score_threshold,
                scene_index_id=scene_index_id
            )

            self.output_message.push_update()
            video_content = VideoContent(
                status=MsgStatus.progress,
                status_message="Started video compilation",
                agent_name=self.agent_name,
            )

            self.output_message.content.append(video_content)

            search_summary_content = TextContent(
                status=MsgStatus.progress,
                status_message="Started generating summary of search results",
                agent_name=self.agent_name,
            )

            self.output_message.content.append(search_summary_content)

            shots = search_results.get_shots()

            if not shots:
                search_result_content.status = MsgStatus.error
                search_result_content.status_message = "Failed to get search results"
                
                video_content.status = MsgStatus.error
                video_content.status_message = "Couldn't compile a video of the results"
                
                
                search_summary_content.status = MsgStatus.error
                search_summary_content.status_message = "Failed to generate summary of results"
                

                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message=f"Failed due to no search results found for query {query}",
                        data={
                        "message": f"Failed due to no search results found for query {query}",
                    },
                )
                
            shots_dict_list = [
                {
                    "search_score": shot["search_score"],
                    "start": shot["start"],
                    "end": shot["end"],
                    "text": shot["text"],
                }
                for shot in shots
            ]
                
            search_results_shots = {
                "video_id": video_id,
                "video_title": video.get("title"),
                "stream_url": video.get("stream_url"),
                "duration": video.get("length"),
                "shots": shots_dict_list
            }

            search_result_content.search_results = [
                SearchData(
                    video_id=search_results_shots["video_id"],
                    video_title=search_results_shots["video_title"],
                    stream_url=search_results_shots["stream_url"],
                    duration=search_results_shots["duration"],
                    shots=[ShotData(**shot) for shot in search_results_shots["shots"]]
                )
            ]

            search_result_content.status = MsgStatus.success
            search_result_content.status_message = "Searching in footage done"


            self.output_message.actions.append("Generating search result and footage clip..")
            self.output_message.push_update()

            compilation_stream_url = search_results.compile()
            video_content.video = VideoData(stream_url=compilation_stream_url)
            video_content.status = MsgStatus.success
            video_content.status_message = "Footage video compiled"

            self.output_message.actions.append("Generating search result summary..")
            self.output_message.push_update()

            search_result_text_list = [shot.text for shot in shots]
            search_result_text = "\n\n".join(search_result_text_list)

            search_summary_content.text = search_result_text
            search_summary_content.status = MsgStatus.success
            search_summary_content.status_message = "Here is the summary of search results."

            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")

            if search_result_content.status != MsgStatus.success:
                search_result_content.status = MsgStatus.error
                search_result_content.status_message = "Failed to get search results"

            if video_content.status != MsgStatus.success:
                video_content.status = MsgStatus.error
                video_content.status_message = (
                    "Failed to create compilation of search results"
                )

            if search_summary_content.status != MsgStatus.success:
                search_summary_content.status = MsgStatus.error
                search_summary_content.status_message = (
                    "Failed to generate summary of results"
                )

            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Failed to search: {e}",
            )
        
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed task successfully!",
            data={
                "stream_url": compilation_stream_url,
                "text": search_result_text
            },
        )
